_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    # '../_base_/datasets/ssl_lvis_v0.5_instance.py',
    '../_base_/datasets/lvis_v0.5_instance_ssl.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='FPN'
        ),
    roi_head=dict(
        type='SSLRoIHead',
        # type='StandardRoIHead',
        bbox_head=dict(
                        # type='Shared2FCBBoxHead',
                        type='SSLShared2FCBBoxHead',
                       num_classes=1230),
        mask_head=dict(num_classes=1230)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                    (1333, 768), (1333, 800)],
#         multiscale_mode='value',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
#
# data = dict(samples_per_gpu=2,
#             workers_per_gpu=4,
#             train=dict(dataset=dict(pipeline=train_pipeline)))

evaluation = dict(interval=1, metric=['bbox'])
load_from = '/data1/PycharmProjects/dcl/long-tail-detection/baseline/work_dirs/pretrain_4x/latest.pth'

find_unused_parameters = True