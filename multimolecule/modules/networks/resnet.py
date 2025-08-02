# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from typing import Callable, Type

from torch import Tensor, nn
from transformers.activations import ACT2FN

from ..normlizations import LayerNorm2d
from .registry import NETWORKS


@NETWORKS.register("resnet")
class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_channels: int | None = None,
        num_labels: int = 1,
        block: Type[BasicBlock | BottleneckBlock] | str = "auto",
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(block, str):
            block = block.lower()
            if block == "auto":
                block = BasicBlock if num_layers < 6 else BottleneckBlock
            elif block in ("basic", "basicblock"):
                block = BasicBlock
            elif block in ("bottleneck", "bottleneckblock"):
                block = BottleneckBlock
            else:
                raise ValueError(f"Unable to resolve block type: {block}. Please use 'auto', 'basic', or 'bottleneck'.")
        if num_channels is None:
            num_channels = hidden_size // 8
        if normalization is None:
            normalization = LayerNorm2d

        self.projection = conv1x1(hidden_size, num_channels)
        self.norm = normalization(num_channels)
        self.activation = ACT2FN[activation]
        self.layers = nn.Sequential(
            *[block(num_channels, normalization=normalization, activation=activation) for _ in range(num_layers)]
        )
        self.prediction = nn.Linear(num_channels, num_labels)
        self.nonlinearity = activation

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nonlinearity = (
                    self.nonlinearity if self.nonlinearity in ("relu", "leaky_relu", "tanh", "selu") else "relu"
                )
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                if isinstance(m, BottleneckBlock) and m.norm3.weight is not None:
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.norm2.weight is not None:
                    nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x.transpose(1, 3))
        x = self.norm(x)
        x = self.activation(x)
        x = self.layers(x)
        x = self.prediction(x.transpose(1, 3))
        return x


class BottleneckBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        dilation: int = 1,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = max(out_channels, in_channels) // 4
        if normalization is None:
            normalization = LayerNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, hidden_channels)
        self.norm1 = normalization(hidden_channels)
        self.conv2 = conv3x3(hidden_channels, hidden_channels, stride, groups, dilation)
        self.norm2 = normalization(hidden_channels)
        self.conv3 = conv1x1(hidden_channels, out_channels)
        self.norm3 = normalization(out_channels)
        self.activation = ACT2FN[activation]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        dilation: int = 1,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = max(out_channels, in_channels) // 4
        if normalization is None:
            normalization = LayerNorm2d
        if dilation > 1:
            raise NotImplementedError(f"{self.__class__.__name__} does not support dilation > 1")
        if groups != 1:
            raise NotImplementedError(f"{self.__class__.__name__} only supports groups=1")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, hidden_channels, stride)
        self.norm1 = normalization(hidden_channels)
        self.activation = ACT2FN[activation]
        self.conv2 = conv3x3(hidden_channels, out_channels)
        self.norm2 = normalization(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


def conv3x3(
    in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
