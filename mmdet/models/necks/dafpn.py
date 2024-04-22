# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


###################################################################
# ################## CBAM Spatial Attention  ######################
###################################################################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 3, 1, 1),
                                    nn.GroupNorm(num_groups=32, num_channels=self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(x)
        FM = cab + sab

        return FM


###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.mask = nn.Sequential(nn.Conv2d(self.channel1, 1, 7, 1, 3), nn.ReLU())
        self.high_feature = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 3, 1, 1),
                                          nn.GroupNorm(num_groups=32, num_channels=self.channel1),
                                          nn.ReLU())
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # y: higher-level features
        # in_map: higher-level c
        input_map = self.high_feature(y)
        up = F.interpolate(input_map, size=x.size()[2:], mode='bilinear', align_corners=True)
        mask = self.sa(up)
        mask = self.sigmoid(mask)

        f_feature = x * mask
        b_feature = x * (1 - mask)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = (self.beta * fn) + x + up
        refine1 = self.gn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 - (self.alpha * fp)
        refine2 = self.gn2(refine2)
        refine2 = self.relu2(refine2)

        return refine2


@NECKS.register_module()
class DAFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False):
        super(DAFPN, self).__init__()

        _, C3_size, C4_size, C5_size = in_channels
        feature_size = out_channels

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6_1 = nn.Sequential(nn.Conv2d(C5_size, 512, kernel_size=3, stride=2, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=512),
                                  nn.ReLU())
        # self.positioning = Positioning(512)
        self.P6_2 = nn.Conv2d(512, feature_size, kernel_size=3, stride=1, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(512, feature_size, kernel_size=3, stride=2, padding=1))

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Sequential(nn.Conv2d(C5_size, 512, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=512),
                                  nn.ReLU())
        self.focus3 = Focus(512, 512)
        self.P5_2 = nn.Conv2d(512, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Sequential(nn.Conv2d(C4_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.focus2 = Focus(256, 512)
        self.P4_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Sequential(nn.Conv2d(C3_size, 128, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=128),
                                  nn.ReLU())
        self.focus1 = Focus(128, 256)
        self.P3_2 = nn.Conv2d(128, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P6_x = self.P6_1(C5)
        # P6_x = self.positioning(P6_x)

        P5_x = self.P5_1(C5)
        P5_x = self.focus3(P5_x, P6_x)

        P4_x = self.P4_1(C4)
        P4_x = self.focus2(P4_x, P5_x)

        P3_x = self.P3_1(C3)
        P3_x = self.focus1(P3_x, P4_x)

        P7_x = self.P7_1(P6_x)
        P6_x = self.P6_2(P6_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_2(P3_x)

        outs = [P3_x, P4_x, P5_x, P6_x, P7_x]
        return tuple(outs)
