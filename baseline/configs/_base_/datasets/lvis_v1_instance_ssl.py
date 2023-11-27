# dataset settings
train_dataset_type = 'MultiViewDataset'
test_dataset_type = 'LVISV1Dataset'
_base_ = 'coco_instance.py'
data_root = '/data1/dataset/LVIS/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
]

# base_pipeline = [
#     dict(
#         type='Resize',
#         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                     (1333, 768), (1333, 800)],
#         multiscale_mode='value',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
# ]
#
# sim_pipline = [
#     dict(
#         type='RandomGrayscale',
#         prob=0.2,
#         keep_channels=True,
#         channel_weights=(0.114, 0.587, 0.2989)),
#     dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
#     dict(type='RandomSolarize', prob=0.2),
#     dict(
#         type='ColorJitter_',
#         prob=0.8,
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.1),
# ]
# last_pipline = [
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]

train_pipeline1 = [
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

train_pipeline2 = [
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomSolarize', prob=0.2),
    dict(
        type='ColorJitter_',
        prob=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=102,
    #      contrast_range=(0.6, 1.4),
    #      saturation_range=(0.6, 1.4),
    #      hue_delta=0),
    dict(
        type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Identity'),
    #         dict(type='AutoContrast'),
    #         dict(type='RandEqualize'),
    #         dict(type='RandSolarize'),
    #         dict(type='RandColor'),
    #         dict(type='RandContrast'),
    #         dict(type='RandBrightness'),
    #         dict(type='RandSharpness'),
    #         dict(type='RandPosterize')
    #     ]),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=train_dataset_type,
        # type='ClassBalancedDataset',
        oversample_thr=1e-3,
        num_views=2,
        pipelines=[train_pipeline1, train_pipeline2],
        dataset=dict(
            type='LVISV1Dataset',
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root,
            # classes=classes,
            pipeline=load_pipeline)),
    val=dict(
        type=test_dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),

    test=dict(
        type=test_dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
