_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    # '../_base_/datasets/ssl_lvis_v1_instance.py',
    '../_base_/datasets/lvis_v1_instance_ssl.py',
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
            num_classes=1203,
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
            ),
        mask_head=dict(num_classes=1203, predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001,
    step=[16, 22])

data = dict(samples_per_gpu=8,
            workers_per_gpu=4)
#             train=dict(dataset=dict(pipeline=train_pipeline)))

# workflow = [('train',1),('val', 1)]
evaluation = dict(metric=['bbox'], interval=6)
find_unused_parameters = True
