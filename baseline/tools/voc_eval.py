from argparse import ArgumentParser
import os
import os.path as osp
import time
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

from mmdet import datasets
from mmdet.core import eval_map
import mmcv
import torch
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         get_root_logger, replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4, logger=None):
    det_results = mmcv.load(result_file)
    det_res = list()
    for det in det_results:
        det_res.append(det[0])
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_res,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger=logger,
        nproc=nproc)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('--result', default='./ecmloss_v1.pkl', help='result file path')
    parser.add_argument('--config', default='./work_dirs/ECM_Lvisv1/lvis_v1_r50_ecm_1x.py',
                        help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--nproc',
        type=int,
        default=1,
        help='Processes to be used for computing mAP')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    rank, _ = get_dist_info()

    # build the dataloader
    print(cfg.data.test)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    log_file = osp.join(cfg.work_dir, 'map_ecmloss_v1.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    voc_eval(args.result, test_dataset, args.iou_thr, args.nproc, logger=logger)


if __name__ == '__main__':
    main()