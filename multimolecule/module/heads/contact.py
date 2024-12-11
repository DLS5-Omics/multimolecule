# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Callable, Mapping, Tuple, Type

import torch
from danling import NestedTensor
from danling.modules import MLP
from lazy_imports import try_import
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput
from typing_extensions import TYPE_CHECKING

from .config import HeadConfig
from .generic import PredictionHead
from .output import HeadOutput
from .registry import HeadRegistry

with try_import() as tv:
    from torchvision.models.resnet import BasicBlock, Bottleneck

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.contact.register("simple", default=True)
class ContactHead(PredictionHead):

    output_name: str = "last_hidden_state"

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        out_channels: int = self.config.hidden_size  # type: ignore[assignment]
        self.qk_proj = nn.Linear(out_channels, 2 * out_channels)
        self.ffn = MLP(1, out_channels, residual=False)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping | Tuple[Tensor, ...],
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
            attention_mask = self._get_attention_mask(input_ids)
        output = output * attention_mask.unsqueeze(-1)
        output, _, _ = self._remove_special_tokens(output, attention_mask, input_ids)

        q, k = self.qk_proj(output).chunk(2, dim=-1)
        contact_map = (q @ k.transpose(-2, -1)).unsqueeze(-1)
        contact_map = contact_map + self.ffn(contact_map)
        if "continuous" in outputs:
            contact_map = contact_map * (1 + outputs["continuous"].unsqueeze(dim=-1))  # type: ignore[call-overload]
        return super().forward(contact_map, labels)


@HeadRegistry.contact.register("attention")
class ContactPredictionHead(PredictionHead):
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

    requires_attention: bool = True

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.config.hidden_size = config.num_hidden_layers * config.num_attention_heads
        num_layers = self.config.get("num_layers", 16)
        num_channels = self.config.get("num_channels", self.config.hidden_size // 10)  # type: ignore[operator]
        block = self.config.get("block", "auto")
        self.decoder = ResNet(
            num_layers=num_layers,
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=block,
            num_channels=num_channels,
            num_labels=self.num_labels,
        )
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

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

        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[-1]
        attentions = torch.stack(output, 1)

        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)
        attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        attentions = attentions * attention_mask[:, None, None, :, :]

        # remove cls token attentions
        if self.bos_token_id is not None:
            attentions = attentions[..., 1:, 1:]
            attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        # remove eos token attentions
        if self.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.eos_token_id).to(attentions)
            else:
                last_valid_indices = attention_mask.sum(dim=-1)
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=attentions.device).unsqueeze(0) == last_valid_indices
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]

        # features: batch x channels x input_ids x input_ids (symmetric)
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = attentions.to(self.decoder.proj.weight.device)
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1).squeeze(3)

        return super().forward(attentions, labels, **kwargs)


@HeadRegistry.contact.register("logits")
class ContactLogitsHead(PredictionHead):
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

    requires_attention: bool = False

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        num_layers = self.config.get("num_layers", 16)
        num_channels = self.config.get("num_channels", self.config.hidden_size // 10)  # type: ignore[operator]
        block = self.config.get("block", "auto")
        self.decoder = ResNet(
            num_layers=num_layers,
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=block,
            num_channels=num_channels,
            num_labels=self.num_labels,
        )
        if head_config is not None and head_config.output_name is not None:
            self.output_name = head_config.output_name

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
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[output_name or self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self._get_attention_mask(input_ids)
        output = output * attention_mask.unsqueeze(-1)
        output, _, _ = self._remove_special_tokens(output, attention_mask, input_ids)

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


class LayerNorm2D(nn.GroupNorm):
    def __init__(self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1)
        self.num_channels = num_features

    def __repr__(self):
        return f"{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps}, affine={self.affine})"


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized
