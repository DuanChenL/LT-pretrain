# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV05Dataset'
# data_root = '/data/xuhy/baseline/configs/data/lvis_v1/'
data_root = '/data1/dataset/LVIS/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v0.5_train_5k.json',
            img_prefix=data_root + 'train2017/')),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val_500.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v0.5_val_500.json',
        img_prefix=data_root + 'val2017/'))
evaluation = dict(metric=['bbox'])
