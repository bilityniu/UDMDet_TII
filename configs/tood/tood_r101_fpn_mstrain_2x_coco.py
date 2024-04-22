_base_ = './udmdet_tood_r50_fpn_anchor_based_1x_duo.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
