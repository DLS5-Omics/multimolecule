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

from typing import Callable, Mapping, Tuple, Type

import torch
from danling import NestedTensor
from lazy_imports import try_import
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from typing_extensions import TYPE_CHECKING

from ..criterions import CriterionRegistry
from ..normlizations import LayerNorm2D
from .config import HeadConfig
from .generic import BasePredictionHead, PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .transform import HeadTransformRegistryHF

with try_import() as tv:
    from torchvision.models.resnet import BasicBlock, Bottleneck

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.contact.logits.register("projection", default=True)
class ContactPredictionHead(BasePredictionHead):

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

    require_attentions: bool = False
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        out_channels: int = self.config.hidden_size  # type: ignore[assignment]
        self.q_proj = nn.Linear(out_channels, out_channels)
        self.decoder = nn.Linear(out_channels, self.num_labels, bias=False)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = CriterionRegistry.build(self.config)
        # self.ffn = MLP(1, out_channels, residual=False)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, _, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        output = self.dropout(output)
        output = self.transform(output)
        q = self.q_proj(output)
        contact_map = q.unsqueeze(1) * q.unsqueeze(2)
        # contact_map = (q @ q.transpose(-1, -2)).unsqueeze(-1)
        # contact_map = contact_map + self.ffn(contact_map)

        output = self.decoder(contact_map)
        if self.activation is not None:
            output = self.activation(output)
        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(output, Tensor):
                    output = labels.nested_like(output, strict=False)
                return HeadOutput(output, self.criterion(output.concat, labels.concat))
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)


HeadRegistry.contact.setattr("default", ContactPredictionHead)


@HeadRegistry.contact.attention.register("linear")
class ContactAttentionLinearHead(PredictionHead):
    r"""
    Head for tasks in contact-level.

    Performs symmetrization, and average product correct.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "attentions"
    r"""The default output to use for the head."""

    require_attentions: bool = True
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        if head_config is None:
            head_config = HeadConfig(hidden_size=config.num_hidden_layers * config.num_attention_heads)
        else:
            head_config.hidden_size = config.num_hidden_layers * config.num_attention_heads
        super().__init__(config, head_config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the ContactPredictionHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if attention_mask is None:
            if isinstance(input_ids, NestedTensor):
                input_ids, attention_mask = input_ids.tensor, input_ids.mask
            else:
                if input_ids is None:
                    raise ValueError(
                        f"Either attention_mask or input_ids must be provided for {self.__class__.__name__} to work."
                    )
                if self.pad_token_id is None:
                    raise ValueError(
                        f"pad_token_id must be provided when attention_mask is not passed to {self.__class__.__name__}."
                    )
                attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[-1]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        attentions = torch.stack(output, 1).flatten(1, 2).permute(0, 2, 3, 1)

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        attentions, _, _ = self.remove_special_tokens_2d(attentions, attention_mask, input_ids)

        attentions = attentions.to(self.decoder.weight.device)
        attentions = average_product_correct(symmetrize(attentions))

        return super().forward(attentions, labels, **kwargs)


@HeadRegistry.contact.attention.register("resnet")
class ContactAttentionResnetHead(PredictionHead):
    r"""
    Head for tasks in contact-level.

    Performs symmetrization, and average product correct.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "attentions"
    r"""The default output to use for the head."""

    require_attentions: bool = True
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        if head_config is None:
            head_config = HeadConfig(hidden_size=config.num_hidden_layers * config.num_attention_heads)
        else:
            head_config.hidden_size = config.num_hidden_layers * config.num_attention_heads
        super().__init__(config, head_config)
        num_layers = self.config.get("num_layers", 16)
        num_channels = self.config.get("num_channels", self.config.hidden_size)  # type: ignore[operator]
        block = self.config.get("block", "auto")
        self.decoder = ResNet(
            num_layers=num_layers,
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=block,
            num_channels=num_channels,
            num_labels=self.num_labels,
        )

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the ContactPredictionHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[-1]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        attentions = torch.stack(output, 1)

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        attentions, _, _ = self.remove_special_tokens(attentions, attention_mask, input_ids)

        # features: batch x channels x input_ids x input_ids (symmetric)
        batch_size, layers, heads, seq_len, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seq_len, seq_len)
        attentions = attentions.to(self.decoder.proj.weight.device)
        attentions = average_product_correct(symmetrize(attentions))

        return super().forward(attentions, labels, **kwargs)


@HeadRegistry.contact.logits.register("resnet")
class ContactLogitsResnetHead(PredictionHead):
    r"""
    Head for tasks in contact-level.

    Performs symmetrization, and average product correct.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.
    """

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

    require_attentions: bool = False
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        num_layers = self.config.get("num_layers", 16)
        num_channels = self.config.get("num_channels", self.config.hidden_size)  # type: ignore[operator]
        block = self.config.get("block", "auto")
        self.decoder = ResNet(
            num_layers=num_layers,
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=block,
            num_channels=num_channels,
            num_labels=self.num_labels,
        )

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **kwargs,
    ) -> HeadOutput:
        r"""
        Forward pass of the ContactPredictionHead.

        Args:
            outputs: The outputs of the model.
            attention_mask: The attention mask for the inputs.
            input_ids: The input ids for the inputs.
            labels: The labels for the head.
            output_name: The name of the output to use.
                Defaults to `self.output_name`.
        """
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, _, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        # make symmetric contact map
        contact_map = output.unsqueeze(1) * output.unsqueeze(2)

        return super().forward(contact_map, labels, **kwargs)


class ResNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        block: Type[BasicBlock | Bottleneck] | str = "auto",
        num_channels: int | None = None,
        num_labels: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        zero_init_residual: bool = True,
    ) -> None:
        tv.check()
        super().__init__()

        if block == "auto":
            block = BasicBlock if num_layers < 50 else Bottleneck
        elif block in ("basic", "BasicBlock"):
            block = BasicBlock
        elif block in ("bottleneck", "Bottleneck"):
            block = Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block}")
        if num_channels is None:
            num_channels = hidden_size // 10
        if norm_layer is None:
            norm_layer = LayerNorm2D

        self.proj = nn.Conv2d(hidden_size, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = norm_layer(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            *[block(num_channels, num_channels, norm_layer=norm_layer) for _ in range(num_layers)]  # type: ignore
        )
        self.output = nn.Linear(num_channels, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x.transpose(1, 3))
        x = self.norm(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.output(x.transpose(1, 3))
        return x


def symmetrize(x: Tensor) -> Tensor:
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(1, 2)


def average_product_correct(x: Tensor) -> Tensor:
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(1, keepdims=True)
    a2 = x.sum(2, keepdims=True)
    a12 = x.sum((1, 2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized
