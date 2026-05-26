# --------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import DCNv4Function

class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

class DCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, shape=None):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)


        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, L, -1)

        if not self.without_pointwise:
            x = self.output_proj(x)
        return x
    

class DCNv4_sep(DCNv4):
    """
    DCNv4_sep: A variant of DCNv4 where offsets and masks are computed
    using separate feature inputs.
    """
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=1,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        :param channels: Number of input/output channels.
        :param kernel_size: Kernel size for convolution.
        :param stride: Stride of the convolution.
        :param pad: Padding size.
        :param dilation: Dilation rate.
        :param group: Number of groups for group convolution.
        :param offset_scale: Scaling factor for offsets.
        :param dw_kernel_size: Depthwise kernel size for computing offsets and masks.
        :param center_feature_scale: Enable center feature scaling.
        :param remove_center: Remove center points in convolution.
        :param output_bias: Whether to use bias in the output projection.
        :param without_pointwise: Skip pointwise projections.
        """
        super().__init__(channels, kernel_size, stride, pad, dilation, group,
                         offset_scale, dw_kernel_size, center_feature_scale,
                         remove_center, output_bias, without_pointwise, **kwargs)

        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))

    def forward(self, input, feature):
        """
        :param input: 변형 컨볼루션을 위한 입력 특징 맵 (shape: [N, C, H, W]).
        :param feature: 오프셋과 마스크를 생성하기 위한 입력 특징 맵 (shape: [N, C, H, W]).
        :return: 변형 컨볼루션의 결과 (shape: [N, C, H, W]).
        """
        # 4D 텐서인 경우 확인
        if len(input.shape) == 4:
            N, C, H, W = input.shape
        else:
            raise ValueError(f"Expected 4D input tensor, but got {input.shape}")

        # Depthwise Convolution으로 feature 처리 (옵션)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(feature)
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).reshape(N * H * W, C)
        else:
            offset_mask_input = feature.permute(0, 2, 3, 1).reshape(N * H * W, C)

        # Linear Layer로 offset 및 mask 계산
        offset_mask_output = self.offset_mask(offset_mask_input)

        # Offset과 Mask로 분리
        offset_mask_output = offset_mask_output.reshape(N, H, W, -1)
        offset, mask = torch.split(offset_mask_output, [2 * self.K, self.K], dim=-1)
        offset = offset.reshape(N, H, W, self.K, 2)
        mask = torch.sigmoid(mask).reshape(N, H, W, self.K)

        # 변형 컨볼루션 수행
        output = DCNv4Function.apply(
            input,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            self.remove_center
        )

        # 결과 반환
        return output
