import math
from copy import deepcopy
import inspect

from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS, PIPELINES, build_dataset
from mmdet.datasets.pipelines import Compose
from torchvision import transforms as _transforms


# register all existing transforms in torchvision
_INCLUDED_TRANSFORMS = ['ColorJitter']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] in _INCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@DATASETS.register_module()
class MultiViewLvisDataset:
    def __init__(self, dataset, oversample_thr, num_views, pipelines, filter_empty_gt=True):
        assert num_views == len(pipelines)
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES
        self.PALETTE = getattr(dataset, 'PALETTE', None)

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)
        # processing multi_views pipeline
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        category_bbox_freq = defaultdict(int)
        bbox_nums = 0
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1

            cat_ids_bbox = self.dataset.get_cat_ids(idx)
            if len(cat_ids_bbox) == 0 and not self.filter_empty_gt:
                cat_ids_bbox = set([len(self.CLASSES)])
            for cat_id in cat_ids_bbox:
                category_bbox_freq[cat_id] += 1
                bbox_nums += 1
        for k, v in category_bbox_freq.items():
            category_bbox_freq[k] = v / bbox_nums
        for k, v in category_freq.items():
            category_freq[k] = math.sqrt((v / num_images) * category_bbox_freq[k])

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        results = self.dataset[ori_index]
        return list(map(lambda pipeline: pipeline(deepcopy(results)), self.pipelines))

    def get_ann_info(self, idx):
        """Get annotation of dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ori_index = self.repeat_indices[idx]
        return self.dataset.get_ann_info(ori_index)

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)