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
from contextlib import nullcontext
from typing import Any, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import HeadConfig, TokenPredictionHead, preserve_batch_norm_stats

from ..modeling_outputs import TokenPredictorOutput
from .configuration_basenji import BasenjiConfig


class BasenjiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BasenjiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["BasenjiBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class BasenjiModel(BasenjiPreTrainedModel):
    """
    The bare Basenji2 backbone. Consumes a long DNA window and returns binned hidden states.

    The architecture faithfully reproduces the upstream Basenji2 trunk: a pre-activation
    convolution stem (`GELU -> Conv -> BatchNorm -> MaxPool`), a width-growing reducing tower, a
    dilated residual tower on a wide stream with a narrow bottleneck, a `Cropping1D`, and a final
    pointwise convolution block. The positional axis of the output is *binned*: a window of
    `config.sequence_length` base pairs is downsampled by the stem/tower and cropped, so
    `last_hidden_state` has shape `(batch_size, num_bins, head_hidden_size)`.

    Examples:
        >>> from multimolecule import BasenjiConfig, BasenjiModel
        >>> config = BasenjiConfig(
        ...     sequence_length=256, stem_channels=8, conv_tower_channels=[8],
        ...     stem_pool_size=2, head_hidden_size=8, crop_bins=2,
        ...     blocks={"num_blocks": 1, "kernel_size": 3, "bottleneck_size": 4},
        ... )
        >>> model = BasenjiModel(config)
        >>> import torch
        >>> input_ids = torch.randint(config.vocab_size, (1, 256))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 60, 8])
    """

    def __init__(self, config: BasenjiConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embeddings = BasenjiEmbedding(config)
        self.encoder = BasenjiEncoder(config)
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
    ) -> BaseModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
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
        encoder_outputs = self.encoder(embedding_output, **kwargs)

        return BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class BasenjiForTokenPrediction(BasenjiPreTrainedModel):
    """
    Basenji2 with a pointwise regression head over genomic coverage tracks.

    The binned positional axis is treated as the "token" axis: logits have shape
    `(batch_size, num_bins, num_labels)` where `num_labels` is the number of coverage tracks.

    Examples:
        >>> import torch
        >>> from multimolecule import BasenjiConfig, BasenjiForTokenPrediction
        >>> config = BasenjiConfig(
        ...     sequence_length=256, stem_channels=8, conv_tower_channels=[8],
        ...     stem_pool_size=2, head_hidden_size=8, crop_bins=2, num_labels=4,
        ...     blocks={"num_blocks": 1, "kernel_size": 3, "bottleneck_size": 4},
        ... )
        >>> model = BasenjiForTokenPrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 256))
        >>> output = model(input_ids, labels=torch.randn(1, 60, 4))
        >>> output["logits"].shape
        torch.Size([1, 60, 4])
    """

    def __init__(self, config: BasenjiConfig):
        super().__init__(config)
        self.model = BasenjiModel(config)
        # The shared TokenPredictionHead is the upstream `Dense(head_hidden_size -> num_labels)`
        # final layer: an identity transform with a biased linear decoder. Upstream applies a
        # `softplus` activation on the Dense output; `softplus` is not part of `ACT2FN`, so it is
        # applied explicitly in `forward` below (the model's output transform) to keep parity
        # while reusing the shared head unchanged.
        token_head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if token_head_config.num_labels is None:
            token_head_config.num_labels = config.num_labels
        if token_head_config.hidden_size is None:
            token_head_config.hidden_size = config.head_hidden_size
        if token_head_config.problem_type is None:
            token_head_config.problem_type = "regression"
        if token_head_config.transform is None:
            token_head_config.transform = None
        if token_head_config.act is None:
            token_head_config.act = None
        self.token_head = TokenPredictionHead(config, token_head_config)
        self.head_config = self.token_head.config
        self.head_act = config.head_act
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        head_outputs = BaseModelOutput(last_hidden_state=outputs.last_hidden_state)
        # The binned axis has no special tokens; pass an all-ones mask so the shared head keeps
        # every bin. The head computes the unactivated upstream `Dense` projection.
        bin_mask = outputs.last_hidden_state.new_ones(outputs.last_hidden_state.shape[:2], dtype=torch.long)
        output = self.token_head(head_outputs, bin_mask, None, None)

        if self.head_act is None:
            logits = output.logits
        elif self.head_act == "softplus":
            logits = F.softplus(output.logits)
        else:
            logits = ACT2FN[self.head_act](output.logits)

        loss = None
        if labels is not None:
            loss = self.token_head.criterion(logits, labels)

        return TokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class BasenjiEmbedding(nn.Module):
    """One-hot feature projection following the MultiMolecule DNA token order."""

    def __init__(self, config: BasenjiConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class BasenjiEncoder(nn.Module):
    """The full Basenji2 trunk: stem, reducing tower, dilated residual tower, crop, and head block."""

    def __init__(self, config: BasenjiConfig):
        super().__init__()
        self.config = config

        # Stem `conv_block`: a pre-activation convolution operating directly on the one-hot DNA.
        self.stem = BasenjiConvLayer(
            config,
            in_channels=config.vocab_size,
            out_channels=config.stem_channels,
            kernel_size=config.stem_kernel_size,
            pool_size=config.stem_pool_size,
        )

        # Reducing `conv_tower`: explicit growing-width schedule, each stage halves resolution.
        tower: list[nn.Module] = []
        in_channels = config.stem_channels
        for out_channels in config.conv_tower_channels:
            tower.append(
                BasenjiConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.conv_tower_kernel_size,
                    pool_size=config.stem_pool_size,
                )
            )
            in_channels = out_channels
        self.conv_tower = nn.ModuleList(tower)

        # Dilated residual tower on the wide stream with a narrow bottleneck.
        block_config = config.blocks
        blocks: list[nn.Module] = []
        dilation = float(block_config["dilation"])
        for _ in range(block_config["num_blocks"]):
            blocks.append(
                BasenjiBlock(
                    config,
                    in_channels=in_channels,
                    bottleneck_size=block_config["bottleneck_size"],
                    kernel_size=block_config["kernel_size"],
                    dilation=max(1, int(round(dilation))),
                )
            )
            dilation *= block_config["dilation_rate"]
            if block_config["round_dilation"]:
                dilation = float(round(dilation))
        self.blocks = nn.ModuleList(blocks)

        self.crop_bins = config.crop_bins

        # Final pointwise `conv_block` feeding the track head.
        self.head = BasenjiConvLayer(
            config,
            in_channels=in_channels,
            out_channels=config.head_hidden_size,
            kernel_size=1,
            dropout=config.hidden_dropout,
        )
        # Upstream applies one more GELU on the head-block output before the `Dense` layer.
        self.act = ACT2FN[config.hidden_act]
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        record_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states) or kwargs.get(
            "output_contexts", self.config.output_contexts
        )
        all_hidden_states: tuple[Tensor, ...] | None = () if record_hidden_states else None

        hidden_state = self.stem(hidden_state)
        if record_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]
        for layer in self.conv_tower:
            hidden_state = layer(hidden_state)
            if record_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_state,
                    context_fn=lambda block=block: (nullcontext(), preserve_batch_norm_stats(block)),
                )
            else:
                hidden_state = block(hidden_state)
            if record_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]

        if self.crop_bins > 0:
            hidden_state = hidden_state[..., self.crop_bins : hidden_state.shape[-1] - self.crop_bins]
        hidden_state = self.head(hidden_state)
        hidden_state = self.act(hidden_state)

        last_hidden_state = hidden_state.transpose(1, 2)
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states)


class BasenjiBlock(nn.Module):
    """
    Dilated residual block (`dilated_residual`).

    Runs on an `in_channels` residual stream: a dilated convolution bottlenecks to
    `bottleneck_size` channels, a pointwise convolution projects back, then a residual add.
    """

    def __init__(
        self,
        config: BasenjiConfig,
        in_channels: int,
        bottleneck_size: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = BasenjiConvLayer(
            config,
            in_channels=in_channels,
            out_channels=bottleneck_size,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = BasenjiConvLayer(
            config,
            in_channels=bottleneck_size,
            out_channels=in_channels,
            kernel_size=1,
            dropout=config.blocks["dropout"],
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class BasenjiConvLayer(nn.Module):
    """
    Basenji2 pre-activation convolution block (`conv_block`).

    Order: `GELU -> Conv1d (bias-free) -> BatchNorm -> [Dropout] -> [MaxPool]`. The convolutions
    are bias-free because batch normalization follows them; the activation precedes the
    convolution (pre-activation), matching upstream `basenji.blocks.conv_block`.
    """

    def __init__(
        self,
        config: BasenjiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        pool_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm1d(out_channels, config.batch_norm_eps, config.batch_norm_momentum)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.pool1 = nn.MaxPool1d(pool_size) if pool_size > 1 else None

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.batch_norm1(hidden_state)
        if self.dropout is not None:
            hidden_state = self.dropout(hidden_state)
        if self.pool1 is not None:
            hidden_state = self.pool1(hidden_state)
        return hidden_state
