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

from typing import Callable, List, Type

from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from ..normlizations import LayerNorm2d
from .registry import NETWORKS
from .resnet import BasicBlock, BottleneckBlock, conv1x1


@NETWORKS.register("unet")
class UNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_channels: int | None = None,
        num_labels: int = 1,
        block: Type[BasicBlock | BottleneckBlock] | str = "auto",
        projection: str = "conv",
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        zero_init_residual: bool = True,
    ) -> None:
        if not num_layers % 2 == 0:
            raise ValueError(f"{self.__class__.__name__} requires an even number of layers, but got {num_layers}.")
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
        self.encoder = Encoder(
            num_layers // 2,
            num_channels,
            block=block,
            projection=projection,
            normalization=normalization,
            activation=activation,
        )
        self.decoder = Decoder(
            num_layers // 2,
            num_channels,
            block=block,
            projection=projection,
            normalization=normalization,
            activation=activation,
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
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.prediction(x.transpose(1, 3))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        block: Type[BasicBlock | BottleneckBlock] = BottleneckBlock,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        projection: str = "conv",
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(EncoderLayer(num_channels, block, normalization, activation, projection))
            num_channels *= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, contact_map: Tensor) -> Tensor:
        output = [contact_map]
        for layer in self.layers:
            contact_map = layer(contact_map)
            output.append(contact_map)
        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        block: Type[BasicBlock | BottleneckBlock] = BottleneckBlock,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        projection: str = "conv",
    ):
        super().__init__()
        if projection == "conv":
            self.projection = nn.Conv2d(num_channels, num_channels * 2, kernel_size=2, stride=2)
        else:
            self.projection = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer = block(num_channels * 2, stride=1, normalization=normalization, activation=activation)

    def forward(self, contact_map: Tensor) -> Tensor:
        return self.layer(self.projection(contact_map))


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        block: Type[BasicBlock | BottleneckBlock] = BottleneckBlock,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        projection: str = "conv",
    ):
        super().__init__()
        layers = []
        num_channels = num_channels * 2**num_layers
        for _ in range(num_layers):
            layers.append(DecoderLayer(num_channels, block, normalization, activation, projection))
            num_channels //= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, encoder_outputs: List[Tensor]) -> Tensor:
        output = encoder_outputs[-1]
        for layer, residual in zip(self.layers, encoder_outputs[-2::-1]):
            output = layer(output, residual)
        return output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        block: Type[BasicBlock | BottleneckBlock] = BottleneckBlock,
        normalization: Callable[..., nn.Module] | None = None,
        activation: str = "relu",
        projection: str = "conv",
    ):
        super().__init__()
        if projection == "conv":
            self.projection = nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=2, stride=2)
        else:
            self.projection = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.layer = block(num_channels // 2, stride=1, normalization=normalization, activation=activation)

    def forward(self, contact_map: Tensor, residual: Tensor) -> Tensor:
        contact_map = self.projection(contact_map)
        diff = residual.shape[3] - contact_map.shape[3]
        left = diff // 2
        right = diff - left
        contact_map = F.pad(contact_map, [left, right, left, right])
        contact_map = contact_map + residual
        return self.layer(contact_map)
