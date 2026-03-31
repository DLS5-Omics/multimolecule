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

        self.supports_batch_process = False

        self.networks = nn.ModuleList(
            [
                SpotRnaNetwork(config, SpotRnaNetworkConfig(**nc) if isinstance(nc, dict) else nc)
                for nc in config.networks
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

        hidden_state = _outer_concatenation(inputs_embeds)
        hidden_state = (hidden_state - self.input_mean.to(hidden_state.device)) / self.input_std.to(hidden_state.device)

        all_logits = [network(hidden_state) for network in self.networks]
        contact_map = torch.stack([torch.sigmoid(l) for l in all_logits]).mean(dim=0)
        logits = torch.stack(all_logits).mean(dim=0)

        loss = None
        if labels is not None:
            L = logits.size(1)
            triu_mask = torch.triu(torch.ones(L, L, device=logits.device, dtype=torch.bool), diagonal=2)
            loss = self.criterion(logits[:, triu_mask], labels[:, triu_mask].float())

        return SpotRnaModelOutput(
            loss=loss,
            logits=logits,
            contact_map=contact_map if not self.training else None,
        )


class SpotRnaNetwork(nn.Module):
    def __init__(self, config: SpotRnaConfig, network_config: SpotRnaNetworkConfig):
        super().__init__()

        # Some checkpoints use per-network input stats on top of the shared ensemble normalization.
        self.register_buffer("input_scale", torch.ones(1, 1, 1, config.input_channels))
        self.register_buffer("input_bias", torch.zeros(1, 1, 1, config.input_channels))
        self.initial_conv = nn.Conv2d(config.input_channels, network_config.conv_channels, kernel_size=3, padding=1)
        self.conv_blocks = nn.ModuleList(
            [SpotRnaConvBlock(config, network_config, block_index=i) for i in range(network_config.num_conv_blocks)]
        )
        self.output_act = ACT2FN[network_config.output_act]
        self.output_norm = SpotRnaLayerNorm(network_config.conv_channels)

        if network_config.num_blstm_blocks > 0:
            self.blstm = SpotRna2dBLSTM(network_config)
            fc_in_dim = network_config.blstm_hidden_size * 4
        else:
            self.blstm = None
            fc_in_dim = network_config.conv_channels

        fc_blocks = []
        for i in range(network_config.num_fc_blocks):
            in_dim = fc_in_dim if i == 0 else network_config.fc_hidden_size
            fc_blocks.append(SpotRnaFCBlock(config, network_config, in_dim, network_config.fc_hidden_size))
        self.fc_blocks = nn.ModuleList(fc_blocks)

        last_dim = network_config.fc_hidden_size if network_config.num_fc_blocks > 0 else fc_in_dim
        self.output_fc = nn.Linear(last_dim, 1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = hidden_state * self.input_scale + self.input_bias

        L = hidden_state.size(1)
        zero_mask = (
            torch.triu(torch.ones(L, L, device=hidden_state.device, dtype=hidden_state.dtype), diagonal=2)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        hidden_state = hidden_state.permute(0, 3, 1, 2)
        hidden_state = self.initial_conv(hidden_state)

        for block in self.conv_blocks:
            hidden_state = block(hidden_state, zero_mask)

        hidden_state = self.output_act(hidden_state)
        hidden_state = self.output_norm(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 3, 1)

        triu_mask = torch.triu(torch.ones(L, L, device=hidden_state.device, dtype=torch.bool), diagonal=2)

        if self.blstm is not None:
            hidden_state = hidden_state * zero_mask[:, 0].unsqueeze(-1)
            hidden_state = hidden_state.permute(0, 3, 1, 2)
            hidden_state = self.blstm(hidden_state)
            hidden_state = hidden_state.flip(dims=[2])

        batch_size = hidden_state.size(0)
        hidden_state = hidden_state[:, triu_mask, :]

        for fc_block in self.fc_blocks:
            hidden_state = fc_block(hidden_state)

        flat_logits = self.output_fc(hidden_state).squeeze(-1)

        logits = torch.full(
            (batch_size, L, L),
            torch.finfo(hidden_state.dtype).min,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        logits[:, triu_mask] = flat_logits
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

    def forward(self, hidden_state: Tensor, zero_mask: Tensor) -> Tensor:
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = hidden_state * zero_mask
        hidden_state = self.conv1(hidden_state)

        hidden_state = self.act2(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = hidden_state * zero_mask
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state + residual


class SpotRna2dBLSTM(nn.LSTM):
    def __init__(self, network_config: SpotRnaNetworkConfig):
        super().__init__(
            input_size=network_config.conv_channels,
            hidden_size=network_config.blstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, hidden_state: Tensor, hx=None):  # type: ignore[override]
        batch, channels, rows, cols = hidden_state.shape
        nhwc = hidden_state.permute(0, 2, 3, 1)

        flipped = nhwc.flip(dims=[2])

        row_input = flipped.reshape(batch * rows, cols, channels)
        row_out = self._forward_packed(row_input, cols, batch, hx)
        row_out = row_out.reshape(batch, rows, cols, -1)

        # The TensorFlow graph reuses the same BLSTM weights on the transposed pair map.
        col_input = flipped.permute(0, 2, 1, 3).reshape(batch * cols, rows, channels)
        col_out = self._forward_packed(col_input, rows, batch, hx)
        col_out = col_out.reshape(batch, cols, rows, -1).permute(0, 2, 1, 3)

        return torch.cat([row_out, col_out], dim=-1)

    def _forward_packed(self, sequence: Tensor, seq_len: int, batch: int, hx=None) -> Tensor:
        lengths = torch.arange(seq_len - 1, -1, -1, device=sequence.device, dtype=torch.long).repeat(batch)
        valid = lengths > 0
        output = sequence.new_zeros(sequence.size(0), sequence.size(1), self.hidden_size * 2)
        if not valid.any():
            return output

        packed = nn.utils.rnn.pack_padded_sequence(
            sequence[valid],
            lengths[valid].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_hx = None
        if hx is not None:
            packed_hx = (hx[0][:, valid], hx[1][:, valid])

        packed_out, _ = super().forward(packed, packed_hx)
        valid_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        output[valid] = valid_out
        return output


class SpotRnaFCBlock(nn.Module):
    def __init__(
        self, config: SpotRnaConfig, network_config: SpotRnaNetworkConfig, in_features: int, out_features: int
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        fc_act = getattr(network_config, "fc_act", None) or network_config.hidden_act
        self.act = ACT2FN[fc_act]
        self.dropout = nn.Dropout(config.fc_dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.fc(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.norm(hidden_state)
        return hidden_state


def _outer_concatenation(one_hot: Tensor) -> Tensor:
    L = one_hot.size(1)
    col = one_hot.unsqueeze(1).expand(-1, L, -1, -1)
    row = one_hot.unsqueeze(2).expand(-1, -1, L, -1)
    return torch.cat([col, row], dim=-1)


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
