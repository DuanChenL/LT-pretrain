_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v0.5_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='FPN'
        ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=1230), mask_head=dict(num_classes=1230)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1333,800)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
]

train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(type='MultiImageMixDataset',
                       dataset=dict(pipeline=load_pipeline),
                       pipeline=train_pipeline))

find_unused_parameters = True