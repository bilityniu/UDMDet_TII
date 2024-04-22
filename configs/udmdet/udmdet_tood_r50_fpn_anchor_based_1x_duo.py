_base_ = './udmdet_tood_r50_fpn_1x_duo.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
