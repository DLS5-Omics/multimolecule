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

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_malinois import MalinoisConfig


class MalinoisPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MalinoisConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["MalinoisConvBlock", "MalinoisBranchedLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, MalinoisGroupedLinear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(3))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class MalinoisModel(MalinoisPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import MalinoisConfig, MalinoisModel, DnaTokenizer
        >>> config = MalinoisConfig()
        >>> model = MalinoisModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/malinois")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 420])
    """

    def __init__(self, config: MalinoisConfig):
        super().__init__(config)
        self.embeddings = MalinoisEmbedding(config)
        self.encoder = MalinoisEncoder(config)
        self.pooler = MalinoisPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MalinoisModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)

        return MalinoisModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class MalinoisForSequencePrediction(MalinoisPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import MalinoisConfig, MalinoisForSequencePrediction, DnaTokenizer
        >>> config = MalinoisConfig()
        >>> model = MalinoisForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/malinois")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input, labels=torch.randn(2, 3))
        >>> output["logits"].shape
        torch.Size([2, 3])
    """

    def __init__(self, config: MalinoisConfig):
        super().__init__(config)
        self.model = MalinoisModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_labels == 3:
            return ["K562", "HepG2", "SK-N-SH"]
        return [f"cell_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MalinoisEmbedding(nn.Module):
    def __init__(self, config: MalinoisConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.input_length = config.input_length
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = cast(Tensor, self._dtype_reference).dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if inputs_embeds.size(-2) != self.input_length:
            raise ValueError(
                f"Malinois expects input length {self.input_length}, got {inputs_embeds.size(-2)}. "
                "Pad candidate sequences with the upstream MPRA plasmid flanks before inference."
            )
        if inputs_embeds.size(-1) != self.vocab_size:
            raise ValueError(f"Malinois expects {self.vocab_size} input channels, got {inputs_embeds.size(-1)}.")
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class MalinoisEncoder(nn.Module):
    def __init__(self, config: MalinoisConfig):
        super().__init__()
        # The first convolution consumes one channel per MultiMolecule tokenizer symbol. Upstream
        # Malinois only has 4 nucleotide channels (A, C, G, T); `convert_checkpoint.py` expands the
        # converted kernel into the MultiMolecule vocabulary order, leaving the extra "N" slot zero.
        in_channels = config.vocab_size
        blocks = []
        # Upstream pooling schedule after the three conv blocks: maxpool(3), maxpool(4), maxpool(4)
        # with a (1, 1) constant pad before the final pool.
        pool_sizes = [3, 4, 4]
        pad_before_pool = [(0, 0), (0, 0), (1, 1)]
        for out_channels, kernel_size, pool_size, pool_pad in zip(
            config.conv_channels, config.conv_kernel_sizes, pool_sizes, pad_before_pool
        ):
            blocks.append(MalinoisConvBlock(config, in_channels, out_channels, kernel_size, pool_size, pool_pad))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return self.flatten(hidden_state)


class MalinoisConvBlock(nn.Module):
    def __init__(
        self,
        config: MalinoisConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        pool_pad: tuple[int, int],
    ):
        super().__init__()
        left, right = _symmetric_padding(kernel_size)
        self.conv_pad = nn.ConstantPad1d((left, right), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(out_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.act = ACT2FN[config.linear_act]
        self.pool_pad = nn.ConstantPad1d(pool_pad, 0.0)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv(self.conv_pad(hidden_state))
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.pool(self.pool_pad(hidden_state))
        return hidden_state


class MalinoisPooler(nn.Module):
    def __init__(self, config: MalinoisConfig):
        super().__init__()
        in_features = config.conv_channels[-1] * config.flatten_factor
        layers: list[nn.Module] = []
        for _ in range(config.num_linear_layers):
            layers.append(MalinoisPoolerLayer(config, in_features, config.linear_channels))
            in_features = config.linear_channels
        self.layers = nn.ModuleList(layers)
        self.branched = MalinoisBranchedTower(config, in_features)

    def forward(self, hidden_state: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return self.branched(hidden_state)


class MalinoisPoolerLayer(nn.Module):
    def __init__(self, config: MalinoisConfig, in_features: int, out_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features, config.batch_norm_eps, config.batch_norm_momentum)
        self.act = ACT2FN[config.linear_act]
        self.dropout = nn.Dropout(config.linear_dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class MalinoisBranchedTower(nn.Module):
    """
    A tower of independent per-cell-line branches. Each of the ``num_labels`` branches owns a private stack of
    grouped linear layers, so the cell lines do not share parameters after the shared convolutional trunk.
    """

    def __init__(self, config: MalinoisConfig, in_features: int):
        super().__init__()
        self.num_branches = config.num_labels
        self.branched_channels = config.branched_channels
        layers = []
        current = in_features
        for index in range(config.num_branched_layers):
            is_last = index + 1 == config.num_branched_layers
            layers.append(
                MalinoisBranchedLayer(
                    config,
                    current,
                    config.branched_channels,
                    self.num_branches,
                    activation=None if is_last else config.branched_act,
                    dropout=0.0 if is_last else config.branched_dropout,
                )
            )
            current = config.branched_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = hidden_state.repeat(1, self.num_branches)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MalinoisBranchedLayer(nn.Module):
    def __init__(
        self,
        config: MalinoisConfig,
        in_features: int,
        out_features: int,
        num_branches: int,
        activation: str | None,
        dropout: float,
    ):
        super().__init__()
        self.linear = MalinoisGroupedLinear(in_features, out_features, num_branches)
        self.act = ACT2FN[activation] if activation is not None else None
        self.dropout = nn.Dropout(dropout) if activation is not None else None

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.linear(hidden_state)
        if self.act is not None:
            hidden_state = self.act(hidden_state)
        if self.dropout is not None:
            hidden_state = self.dropout(hidden_state)
        return hidden_state


class MalinoisGroupedLinear(nn.Module):
    """
    A block-diagonal linear layer that applies an independent ``Linear(in_group_size, out_group_size)`` to each of
    ``groups`` contiguous feature blocks. The flattened input is interpreted in branch-major order: feature block
    ``g`` occupies columns ``[g * in_group_size, (g + 1) * in_group_size)``.
    """

    def __init__(self, in_group_size: int, out_group_size: int, groups: int):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))

    def forward(self, hidden_state: Tensor) -> Tensor:
        batch = hidden_state.shape[0]
        reorg = hidden_state.reshape(batch, self.groups, self.in_group_size).transpose(0, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        return hook.transpose(0, 1).reshape(batch, self.groups * self.out_group_size)


def _symmetric_padding(kernel_size: int) -> tuple[int, int]:
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return max(0, left), max(0, right)


@dataclass
class MalinoisModelOutput(ModelOutput):
    """
    Base class for outputs of the Malinois model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, flattened_conv_features)`):
            Flattened feature map produced by the convolutional encoder.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, num_labels * branched_channels)`):
            Branch-major sequence-level representation produced by the fully-connected and branched tower. The first
            `branched_channels` features belong to the K562 branch, the next to HepG2, and the last to SK-N-SH.
        hidden_states:
            Always `None`; Malinois feature maps are not recorded.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Always `None`; Malinois is a convolutional model without attention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
