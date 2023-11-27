# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetMseWeightHook(Hook):
    """Set runner's epoch information to the model."""
    def __init__(self,
                 train_loader_cfg=None):
        self.train_loader_cfg = train_loader_cfg

    def before_train_epoch(self, runner):
        t = runner.epoch / runner.max_epochs
        from mmdet.datasets import build_dataloader
        if runner.epoch >= 1 and t <= 0.5:
            runner.model.module.loss_mse.loss_weight += 0.2
            runner.data_loader.dataset.dynamic_update(t)
            dataset = runner.data_loader.dataset
            dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
            data_loaders = [build_dataloader(ds, **self.train_loader_cfg) for ds in dataset]
            runner.data_loader = data_loaders[0]
        elif t > 0.5:
            runner.model.module.loss_mse.loss_weight = 2.0
            runner.data_loader.dataset.dynamic_update(0.5)
            dataset = runner.data_loader.dataset
            dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
            data_loaders = [build_dataloader(ds, **self.train_loader_cfg) for ds in dataset]
            runner.data_loader = data_loaders[0]

