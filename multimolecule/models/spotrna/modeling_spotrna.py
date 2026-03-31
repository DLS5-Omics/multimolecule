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
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_spotrna import SpotRnaConfig, SpotRnaNetworkConfig


class SpotRnaPreTrainedModel(PreTrainedModel):

    config_class = SpotRnaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["SpotRnaConvBlock"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.LayerNorm, SpotRnaLayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


class SpotRnaModel(SpotRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import SpotRnaConfig, SpotRnaModel
        >>> config = SpotRnaConfig()
        >>> model = SpotRnaModel(config)
        >>> input_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]])
        >>> output = model(input_ids=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 10, 10])
        >>> output["contact_map"].shape
        torch.Size([1, 10, 10])
    """

    def __init__(self, config: SpotRnaConfig):
        super().__init__(config)

        self.register_buffer(
            "input_mean",
            torch.tensor(
                [0.223542, 0.18919209, 0.26099518, 0.31503478, 0.223542, 0.18919209, 0.26099518, 0.31503478]
            ).reshape(1, 1, 1, 8),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(
                [0.4219779, 0.39735729, 0.44426465, 0.46934235, 0.4219779, 0.39735729, 0.44426465, 0.46934235]
            ).reshape(1, 1, 1, 8),
        )

        self.networks = nn.ModuleList(
            [
                SpotRnaNetwork(
                    config,
                    SpotRnaNetworkConfig(**network_config) if isinstance(network_config, dict) else network_config,
                )
                for network_config in config.networks
            ]
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.post_init()

    def postprocess(self, outputs, input_ids=None, **kwargs):
        return outputs["contact_map"]

    def _prepare_inputs_embeds(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        num_bases = self.config.input_channels // 2
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            canonical_ids = input_ids.clamp(min=0, max=num_bases - 1)
            inputs_embeds = F.one_hot(canonical_ids, num_classes=num_bases).float()
            valid_tokens = (input_ids >= 0) & (input_ids < num_bases)
            inputs_embeds = inputs_embeds * valid_tokens.unsqueeze(-1)
        else:
            if inputs_embeds.size(-1) < num_bases:
                raise ValueError(
                    f"inputs_embeds last dimension ({inputs_embeds.size(-1)}) must be at least {num_bases}."
                )
            inputs_embeds = inputs_embeds[..., :num_bases]

        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return inputs_embeds

    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SpotRnaModelOutput:
        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is not None and isinstance(inputs_embeds, NestedTensor):
            raise TypeError("SpotRnaModel does not support NestedTensor inputs_embeds")
        inputs_embeds = self._prepare_inputs_embeds(input_ids, attention_mask, inputs_embeds)

        hidden_state = _outer_concatenate(inputs_embeds)
        hidden_state = (hidden_state - self.input_mean.to(hidden_state.device)) / self.input_std.to(hidden_state.device)

        ensemble_logits = [network(hidden_state) for network in self.networks]
        contact_map = torch.stack([torch.sigmoid(logits) for logits in ensemble_logits]).mean(dim=0)
        logits = torch.stack(ensemble_logits).mean(dim=0)

        loss = None
        if labels is not None:
            sequence_length = logits.size(1)
            upper_triangle_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=logits.device, dtype=torch.bool), diagonal=2
            )
            loss = self.criterion(logits[:, upper_triangle_mask], labels[:, upper_triangle_mask].float())

        return SpotRnaModelOutput(
            loss=loss,
            logits=logits,
            contact_map=contact_map if not self.training else None,
        )


class SpotRnaNetwork(nn.Module):

    bilstm: SpotRna2DBiLSTM | None

    def __init__(self, config: SpotRnaConfig, network_config: SpotRnaNetworkConfig):
        super().__init__()

        # Some checkpoints use per-network input stats on top of the shared normalization.
        self.register_buffer("input_scale", torch.ones(1, 1, 1, config.input_channels))
        self.register_buffer("input_bias", torch.zeros(1, 1, 1, config.input_channels))
        self.projection = nn.Conv2d(config.input_channels, network_config.conv_channels, kernel_size=3, padding=1)
        self.layers = nn.ModuleList(
            [
                SpotRnaConvBlock(config, network_config, block_index=block_index)
                for block_index in range(network_config.num_conv_blocks)
            ]
        )
        self.activation = ACT2FN[network_config.output_act]
        self.layer_norm = SpotRnaLayerNorm(network_config.conv_channels)

        if network_config.num_blstm_blocks > 0:
            self.bilstm = SpotRna2DBiLSTM(network_config)
            classifier_input_size = network_config.blstm_hidden_size * 4
        else:
            self.bilstm = None
            classifier_input_size = network_config.conv_channels

        classifier = []
        for block_index in range(network_config.num_fc_blocks):
            input_size = classifier_input_size if block_index == 0 else network_config.fc_hidden_size
            classifier.append(SpotRnaFCBlock(config, network_config, input_size, network_config.fc_hidden_size))
        self.classifier = nn.ModuleList(classifier)

        output_size = network_config.fc_hidden_size if network_config.num_fc_blocks > 0 else classifier_input_size
        self.prediction = nn.Linear(output_size, 1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = hidden_state * self.input_scale + self.input_bias

        sequence_length = hidden_state.size(1)
        valid_pair_mask = (
            torch.triu(
                torch.ones(sequence_length, sequence_length, device=hidden_state.device, dtype=hidden_state.dtype),
                diagonal=2,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        hidden_state = hidden_state.permute(0, 3, 1, 2)
        hidden_state = self.projection(hidden_state)

        for layer in self.layers:
            hidden_state = layer(hidden_state, valid_pair_mask)

        hidden_state = self.activation(hidden_state)
        hidden_state = self.layer_norm(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 3, 1)

        upper_triangle_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=hidden_state.device, dtype=torch.bool), diagonal=2
        )

        if self.bilstm is not None:
            hidden_state = hidden_state * valid_pair_mask[:, 0].unsqueeze(-1)
            hidden_state = hidden_state.permute(0, 3, 1, 2)
            hidden_state = self.bilstm(hidden_state)
            hidden_state = hidden_state.flip(dims=[2])

        batch_size = hidden_state.size(0)
        hidden_state = hidden_state[:, upper_triangle_mask, :]

        for layer in self.classifier:
            hidden_state = layer(hidden_state)

        pair_logits = self.prediction(hidden_state).squeeze(-1)

        logits = torch.full(
            (batch_size, sequence_length, sequence_length),
            torch.finfo(hidden_state.dtype).min,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        logits[:, upper_triangle_mask] = pair_logits
        return logits


class SpotRnaLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_state: Tensor) -> Tensor:
        mean = hidden_state.mean(dim=(1, 2, 3), keepdim=True)
        var = hidden_state.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        hidden_state = (hidden_state - mean) / torch.sqrt(var + 1e-12)
        return hidden_state * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SpotRnaConvBlock(nn.Module):
    def __init__(self, config: SpotRnaConfig, network_config: SpotRnaNetworkConfig, block_index: int = 0):
        super().__init__()
        channels = network_config.conv_channels

        # SPOT-RNA cycles dilation per convolution, not per residual block.
        conv3_idx = block_index * 2
        conv5_idx = block_index * 2 + 1
        dilation1 = 2 ** (conv3_idx % network_config.dilation_cycle) if network_config.use_dilation else 1
        dilation2 = 2 ** (conv5_idx % network_config.dilation_cycle) if network_config.use_dilation else 1

        self.act1 = ACT2FN[network_config.hidden_act]
        self.norm1 = SpotRnaLayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation1, dilation=dilation1)
        self.act2 = ACT2FN[network_config.hidden_act]
        self.norm2 = SpotRnaLayerNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2 * dilation2, dilation=dilation2)
        self.dropout = nn.Dropout(config.conv_dropout)

    def forward(self, hidden_state: Tensor, valid_pair_mask: Tensor) -> Tensor:
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = hidden_state * valid_pair_mask
        hidden_state = self.conv1(hidden_state)

        hidden_state = self.act2(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = hidden_state * valid_pair_mask
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state + residual


class SpotRna2DBiLSTM(nn.LSTM):
    def __init__(self, network_config: SpotRnaNetworkConfig):
        super().__init__(
            input_size=network_config.conv_channels,
            hidden_size=network_config.blstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, hidden_state: Tensor, hx=None):  # type: ignore[override]
        batch_size, channels, num_rows, num_cols = hidden_state.shape
        hidden_state = hidden_state.permute(0, 2, 3, 1)

        flipped = hidden_state.flip(dims=[2])

        row_input = flipped.reshape(batch_size * num_rows, num_cols, channels)
        row_output = self._forward_packed(row_input, num_cols, batch_size, hx)
        row_output = row_output.reshape(batch_size, num_rows, num_cols, -1)

        # The TensorFlow graph reuses the same BLSTM weights on the transposed pair map.
        column_input = flipped.permute(0, 2, 1, 3).reshape(batch_size * num_cols, num_rows, channels)
        column_output = self._forward_packed(column_input, num_rows, batch_size, hx)
        column_output = column_output.reshape(batch_size, num_cols, num_rows, -1).permute(0, 2, 1, 3)

        return torch.cat([row_output, column_output], dim=-1)

    def _forward_packed(self, sequence: Tensor, sequence_length: int, batch_size: int, hx=None) -> Tensor:
        lengths = torch.arange(sequence_length - 1, -1, -1, device=sequence.device, dtype=torch.long).repeat(batch_size)
        valid_sequences = lengths > 0
        output = sequence.new_zeros(sequence.size(0), sequence.size(1), self.hidden_size * 2)
        if not valid_sequences.any():
            return output

        packed = nn.utils.rnn.pack_padded_sequence(
            sequence[valid_sequences],
            lengths[valid_sequences].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_hx = None
        if hx is not None:
            packed_hx = (hx[0][:, valid_sequences], hx[1][:, valid_sequences])

        packed_out, _ = super().forward(packed, packed_hx)
        valid_output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=sequence_length)
        output[valid_sequences] = valid_output
        return output


class SpotRnaFCBlock(nn.Module):
    def __init__(
        self, config: SpotRnaConfig, network_config: SpotRnaNetworkConfig, in_features: int, out_features: int
    ):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        hidden_act = getattr(network_config, "fc_act", None) or network_config.hidden_act
        self.activation = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(config.fc_dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layer_norm(hidden_state)
        return hidden_state


def _outer_concatenate(inputs_embeds: Tensor) -> Tensor:
    sequence_length = inputs_embeds.size(1)
    column_features = inputs_embeds.unsqueeze(1).expand(-1, sequence_length, -1, -1)
    row_features = inputs_embeds.unsqueeze(2).expand(-1, -1, sequence_length, -1)
    return torch.cat([column_features, row_features], dim=-1)


@dataclass
class SpotRnaModelOutput(ModelOutput):
    """
    Output type for SPOT-RNA model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Binary cross-entropy loss for base-pair prediction.
        logits (`torch.FloatTensor` of shape `(batch_size, seq_len, seq_len)`):
            Ensemble-averaged prediction logits (before sigmoid).
        contact_map (`torch.FloatTensor` of shape `(batch_size, seq_len, seq_len)`, *optional*):
            Base-pair probability matrix (after sigmoid). Only returned during inference.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contact_map: torch.FloatTensor | None = None
