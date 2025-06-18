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

from typing import Mapping, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from typing_extensions import TYPE_CHECKING

from ..criterions import CriterionRegistry
from ..networks import ResNet, UNet
from .config import HeadConfig
from .generic import BasePredictionHead
from .output import HeadOutput
from .registry import HeadRegistry
from .transform import HeadTransformRegistryHF

if TYPE_CHECKING:
    from multimolecule.models import PreTrainedConfig


@HeadRegistry.contact.logits.register("linear", default=True)
class ContactPredictionHead(BasePredictionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactPredictionHead
        >>> config = PreTrainedConfig(hidden_size=8)
        >>> head = ContactPredictionHead(config)
        >>> input = torch.randn(1, 28, config.hidden_size)
        >>> output = head({"last_hidden_state": input}, attention_mask=torch.ones(1, 28))
    """

    output_name: str = "last_hidden_state"
    r"""The default output to use for the head."""

    require_attentions: bool = False
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        out_channels: int = self.config.hidden_size  # type: ignore[assignment]
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        self.decoder = nn.Linear(out_channels, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = CriterionRegistry.build(self.config)

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
        contact_map = output.unsqueeze(1) * output.unsqueeze(2)
        contact_map = self.decoder(contact_map)
        contact_map = self.symmetrize(contact_map)
        if self.activation is not None:
            contact_map = self.activation(contact_map)

        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(contact_map, Tensor):
                    contact_map = labels.nested_like(contact_map, strict=False)
                return HeadOutput(contact_map, self.criterion(contact_map.concat, labels.concat))
            return HeadOutput(contact_map, self.criterion(contact_map, labels))
        return HeadOutput(contact_map)


@HeadRegistry.contact.logits.register("resnet")
class ContactPredictionResNetHead(ContactPredictionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactPredictionResNetHead
        >>> config = PreTrainedConfig(hidden_size=32)
        >>> head = ContactPredictionResNetHead(config)
        >>> input = torch.randn(1, 28, config.hidden_size)
        >>> output = head({"last_hidden_state": input}, attention_mask=torch.ones(1, 28))
    """

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.decoder = ResNet(
            num_layers=self.config.get("num_layers", 6),
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=self.config.get("block", "auto"),
            num_channels=self.config.get("num_channels"),
            num_labels=self.num_labels,
        )


@HeadRegistry.contact.logits.register("unet")
class ContactPredictionUNetHead(ContactPredictionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactPredictionUNetHead
        >>> config = PreTrainedConfig(hidden_size=32)
        >>> head = ContactPredictionUNetHead(config)
        >>> input = torch.randn(1, 28, config.hidden_size)
        >>> output = head({"last_hidden_state": input}, attention_mask=torch.ones(1, 28))
    """

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.decoder = UNet(
            num_layers=self.config.get("num_layers", 6),
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=self.config.get("block", "auto"),
            num_channels=self.config.get("num_channels"),
            num_labels=self.num_labels,
        )


@HeadRegistry.contact.attention.register("linear")
class ContactAttentionHead(BasePredictionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactAttentionHead
        >>> config = PreTrainedConfig(num_hidden_layers=2, num_attention_heads=4)
        >>> head = ContactAttentionHead(config)
        >>> input = tuple(torch.randn(1, config.num_attention_heads, 28, 28) for _ in range(config.num_hidden_layers))
        >>> output = head({"attentions": input}, attention_mask=torch.ones(1, 28))
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
        self.dropout = nn.Dropout(self.config.dropout)
        self.transform = HeadTransformRegistryHF.build(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=self.config.bias)
        self.activation = ACT2FN[self.config.act] if self.config.act is not None else None
        self.criterion = CriterionRegistry.build(self.config)

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
            output = outputs[-1]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if isinstance(output, (list, tuple)):
            output = torch.stack(output, 1)
        contact_map = output.flatten(1, 2).permute(0, 2, 3, 1)

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        contact_map, _, _ = self.remove_special_tokens_2d(contact_map, attention_mask, input_ids)

        contact_map = self.dropout(contact_map)
        contact_map = self.transform(contact_map)
        contact_map = self.decoder(contact_map)
        contact_map = self.symmetrize(contact_map)
        if self.activation is not None:
            contact_map = self.activation(contact_map)

        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(contact_map, Tensor):
                    contact_map = labels.nested_like(contact_map, strict=False)
                return HeadOutput(contact_map, self.criterion(contact_map.concat, labels.concat))
            return HeadOutput(contact_map, self.criterion(contact_map, labels))
        return HeadOutput(contact_map)


@HeadRegistry.contact.attention.register("resnet")
class ContactAttentionResNetHead(ContactAttentionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactAttentionResNetHead
        >>> config = PreTrainedConfig(num_hidden_layers=8, num_attention_heads=4)
        >>> head = ContactAttentionResNetHead(config)
        >>> input = tuple(torch.randn(1, config.num_attention_heads, 28, 28) for _ in range(config.num_hidden_layers))
        >>> output = head({"attentions": input}, attention_mask=torch.ones(1, 28))
    """

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.decoder = ResNet(
            num_layers=self.config.get("num_layers", 16),
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=self.config.get("block", "auto"),
            num_channels=self.config.get("num_channels"),
            num_labels=self.num_labels,
        )


@HeadRegistry.contact.attention.register("unet")
class ContactAttentionUNetHead(ContactAttentionHead):
    r"""
    Head for tasks in contact-level.

    Args:
        config: The configuration object for the model.
        head_config: The configuration object for the head.
            If None, will use configuration from the `config`.

    Examples:
        >>> import torch
        >>> from multimolecule.models import PreTrainedConfig
        >>> from multimolecule.modules.heads import ContactAttentionUNetHead
        >>> config = PreTrainedConfig(num_hidden_layers=4, num_attention_heads=8)
        >>> head = ContactAttentionUNetHead(config)
        >>> input = tuple(torch.randn(1, config.num_attention_heads, 28, 28) for _ in range(config.num_hidden_layers))
        >>> output = head({"attentions": input}, attention_mask=torch.ones(1, 28))
    """

    output_name: str = "attentions"
    r"""The default output to use for the head."""

    require_attentions: bool = True
    r"""Whether the head requires attentions."""

    def __init__(self, config: PreTrainedConfig, head_config: HeadConfig | None = None):
        super().__init__(config, head_config)
        self.decoder = UNet(
            num_layers=self.config.get("num_layers", 4),
            hidden_size=self.config.hidden_size,  # type: ignore[arg-type]
            block=self.config.get("block", "auto"),
            num_channels=self.config.get("num_channels"),
            num_labels=self.num_labels,
        )


HeadRegistry.contact.setattr("default", ContactAttentionHead)
