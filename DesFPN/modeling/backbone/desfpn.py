# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

__all__ = ["build_resnet_spotfpn_backbone", "SpotFPN"]

class SpotFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum",reduction=16):
        # bottom_up = 是具备{"res2": [56*56*256], "res3":[28*28*512],"res4":[14*14*1024], "res5": [7*7*2048]}的骨干网
        # in_features = ["res2", "res3", "res4", "res5"]
        # out_channels = 256
        super(SpotFPN, self).__init__()
        # 对传入的bottom_up进行断言 判断类型是否是Backbone
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        # 获取bottom_up的输出类型作为FPN的输入类型
        input_shapes = bottom_up.output_shape()
        # f in ["res2", "res3", "res4", "res5"], 获取res块的步长 in_strides = [stride2, stride3, ...]
        in_strides = [input_shapes[f].stride for f in in_features]
        # 获取["res2", "res3", "res4", "res5"]各自的channel = [256， 512， 1025， 2048]
        in_channels = [input_shapes[f].channels for f in in_features]
        # aug_lateral_conv = 最顶层的channel
        aug_lateral_conv = in_channels[-1]
        # 验证strides
        _assert_strides_are_log2_contiguous(in_strides)
        # 横向卷积层
        lateral_convs = []
        # 输出卷积层
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)
            # 设置横向卷积层
            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            # 设置输出卷积层
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            # 卷积层权重初始化参数
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))

            # 在模型中添加上配置信息
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)
            # 将卷积层添加到list中方便计算
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        # 将横向卷积逆序成 [conv(2048*256)...conv(256, 256)]
        self.lateral_convs = lateral_convs[::-1]
        # 将输出卷积逆序 conv(256, 256) ...
        self.output_convs = output_convs[::-1]
        # top_block = None
        self.top_block = top_block
        # self.in_features = ["res2", "res3", "res4", "res5"]
        self.in_features = in_features
        # self.bottom_up = Backbone
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        # self._out_feature_strides.keys() = ["p2", "p3", ... "p6"]
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        # 顶层的步长
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        # 将图片传入骨干网，获取到res对应的 【str:feature map， 。。。】
        bottom_up_features = self.bottom_up(x)
        # 将res2.。。res5变成list
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        # 定义一个输出list
        results = []
        # 将最顶层的feature map 进行SENet
        b, c, _, _ = x[0].size()
        y = self.avg_pool(x[0]).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        # 首先对顶层的res5进行横向卷积
        prev_features_tmp = self.lateral_convs[0](x[0])
        # 将卷积后的结果翻入raw_laternals
        raw_laternals = [prev_features_tmp.clone()]
        # results添加 top-down卷积后的结果
        results.append(self.output_convs[0](prev_features_tmp)+ y)
        
        prev_features = prev_features_tmp 
        # 以上对顶层处理完毕后处理下面的FPN层
        for features, lateral_conv, output_conv in zip(
                x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            # 计算每一层的横向卷积
            lateral_features = lateral_conv(features)
            # 将结果添加到上下卷积list中，插入的list的首位 # M2, M3, M4, M5
            raw_laternals.insert(0, lateral_features.clone())
            # 将两者相加到一起变成向下一层计算的输入纵向
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            # 在results插入计算后的结果 P2, P3, P4, P5
            results.insert(0, output_conv(prev_features))
        # 如果顶层top_block不是空 就进行一步计算
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results) == len(raw_laternals)
        return dict(zip(self._out_features, results)), dict(zip(self._out_features, raw_laternals))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_spotfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = SpotFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,  # LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone