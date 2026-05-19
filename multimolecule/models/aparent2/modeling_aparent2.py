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
from dataclasses import dataclass
from typing import Any, Tuple

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

from multimolecule.modules import Criterion, HeadConfig, preserve_batch_norm_stats

from ..modeling_outputs import SequencePredictorOutput
from .configuration_aparent2 import Aparent2Config


class Aparent2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Aparent2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["Aparent2Block"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.xavier_normal_(module.weight)
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
        elif isinstance(module, Aparent2LibraryBias):
            init.xavier_normal_(module.weight)
            init.zeros_(module.bias)


class Aparent2Model(Aparent2PreTrainedModel):
    """
    The bare APARENT2 residual network.

    APARENT2 predicts a base-pair-resolution cleavage distribution for a fixed 205bp polyadenylation signal window.
    The core hexamer (e.g. ``AATAAA``) is expected to start at position 70 (0-indexed). Variant effect is an
    *input-schema* concern: score a reference and an alternate sequence separately and compare their cleavage /
    isoform predictions; there is no separate ref/alt output dataclass.

    Examples:
        >>> from multimolecule import Aparent2Config, Aparent2Model, DnaTokenizer
        >>> config = Aparent2Config()
        >>> model = Aparent2Model(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/aparent2")
        >>> input = tokenizer("A" * 205, return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 206])
        >>> output["pooler_output"].shape
        torch.Size([1, 206])
    """

    def __init__(self, config: Aparent2Config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.embeddings = Aparent2Embedding(config)
        self.encoder = Aparent2Encoder(config)
        self.prediction = nn.Conv1d(config.hidden_size, 1, kernel_size=1)
        self.library_bias = Aparent2LibraryBias(config)
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
    ) -> Aparent2ModelOutput:
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

        hidden_state = self.encoder(embedding_output, **kwargs)

        sequence_score = self.prediction(hidden_state).squeeze(1)
        logits = self.library_bias(sequence_score)

        return Aparent2ModelOutput(
            logits=logits,
            pooler_output=logits,
            last_hidden_state=hidden_state.transpose(1, 2),
        )


class Aparent2ForSequencePrediction(Aparent2PreTrainedModel):
    """
    APARENT2 with a sequence-level prediction head.

    The backbone already produces a `sequence_length + 1` dimensional cleavage score (the APA cleavage distribution
    before softmax), so this wrapper exposes those converted upstream scores directly and adds the shared
    MultiMolecule regression loss.

    Examples:
        >>> import torch
        >>> from multimolecule import Aparent2Config, Aparent2ForSequencePrediction, DnaTokenizer
        >>> config = Aparent2Config()
        >>> model = Aparent2ForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/aparent2")
        >>> input = tokenizer("A" * 205, return_tensors="pt")
        >>> output = model(**input, labels=torch.randn(1, 206))
        >>> output["logits"].shape
        torch.Size([1, 206])
    """

    def __init__(self, config: Aparent2Config):
        super().__init__(config)
        self.model = Aparent2Model(config)
        head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if head_config.num_labels is None:
            head_config.num_labels = config.num_labels
        if head_config.problem_type is None:
            head_config.problem_type = "regression"
        self.head_config = head_config
        self.criterion = Criterion(head_config)

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
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        loss = self.criterion(logits, labels) if labels is not None else None

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class Aparent2Embedding(nn.Module):
    """One-hot embedding that derives input channels from the MultiMolecule token order."""

    def __init__(self, config: Aparent2Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
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
            valid = (input_ids >= 0) & (input_ids < self.vocab_size)
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype=dtype)
            if not valid.all():
                inputs_embeds = inputs_embeds * valid.unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if inputs_embeds.size(1) != self.sequence_length:
            raise ValueError(
                f"APARENT2 expects exactly {self.sequence_length} input tokens, but got {inputs_embeds.size(1)}."
            )
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class Aparent2Encoder(nn.Module):
    def __init__(self, config: Aparent2Config):
        super().__init__()
        self.config = config
        self.projection = nn.Conv1d(config.vocab_size, config.hidden_size, kernel_size=1)
        self.groups = nn.ModuleList([Aparent2Group(config, dilation=d) for d in config.dilations])
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        hidden_state = self.projection(hidden_state)
        group_skips = []
        for group in self.groups:
            if self.gradient_checkpointing and self.training:
                hidden_state, group_skip = self._gradient_checkpointing_func(
                    group.__call__,
                    hidden_state,
                    context_fn=lambda group=group: (nullcontext(), preserve_batch_norm_stats(group)),
                )
            else:
                hidden_state, group_skip = group(hidden_state)
            group_skips.append(group_skip)
        # Upstream: skip_add = last_block_conv(final_hidden_state) + sum(per-group skip convs).
        skip = self.conv(hidden_state)
        for group_skip in group_skips:
            skip = skip + group_skip
        return skip


class Aparent2Group(nn.Module):
    def __init__(self, config: Aparent2Config, dilation: int = 1):
        super().__init__()
        self.skip = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList([Aparent2Block(config, dilation=dilation) for _ in range(config.num_blocks)])

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.skip(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state, skip


class Aparent2Block(nn.Module):
    def __init__(self, config: Aparent2Config, dilation: int = 1):
        super().__init__()
        momentum = 1.0 - config.batch_norm_momentum
        self.norm1 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, momentum)
        self.act1 = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.norm2 = nn.BatchNorm1d(config.hidden_size, config.batch_norm_eps, momentum)
        self.act2 = ACT2FN[config.hidden_act]
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.kernel_size,
            dilation=dilation,
            padding="same",
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Aparent2LibraryBias(nn.Module):
    """
    Position-wise locally-connected training-sub-library bias.

    The upstream model appends a zero "no-cleavage" position to the per-position score (extending it to
    `sequence_length + 1`), then adds a position-specific affine transform of a one-hot training-sub-library
    indicator. The library indicator is a deterministic constant (the variant-effect workflow always uses
    `library_index`), rebuilt in `forward`; only the locally-connected `weight` and `bias` are learned/persistent.
    """

    def __init__(self, config: Aparent2Config):
        super().__init__()
        self.num_positions = config.sequence_length + 1
        self.num_libraries = config.num_libraries
        self.library_index = config.library_index
        self.weight = nn.Parameter(torch.empty(self.num_positions, config.num_libraries))
        self.bias = nn.Parameter(torch.empty(self.num_positions))

    def forward(self, sequence_score: Tensor) -> Tensor:
        batch_size = sequence_score.size(0)
        zeros = torch.zeros(batch_size, 1, device=sequence_score.device, dtype=sequence_score.dtype)
        extended = torch.cat([sequence_score, zeros], dim=1)
        # Deterministic one-hot training-sub-library indicator, rebuilt every forward.
        library = torch.zeros(self.num_libraries, device=sequence_score.device, dtype=sequence_score.dtype)
        library[self.library_index] = 1.0
        library_bias = self.weight.to(dtype=sequence_score.dtype) @ library + self.bias.to(dtype=sequence_score.dtype)
        return extended + library_bias.unsqueeze(0)


@dataclass
class Aparent2ModelOutput(ModelOutput):
    """
    Base class for outputs of the APARENT2 model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Not produced by the bare model; present for API compatibility.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1)`):
            APA cleavage scores (before SoftMax) for each position plus a trailing "no cleavage in window" bucket.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1)`):
            Same content as `logits`; exposed for sequence-level prediction wrappers.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The residual-network feature map before the final cleavage projection.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states of the model at the output of each layer.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
