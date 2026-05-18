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

import numpy as np
import torch
from danling import NestedTensor
from scipy.interpolate import splev
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import Criterion

from ..modeling_outputs import SequencePredictorOutput
from .configuration_mtsplice import MtSpliceConfig


class MtSplicePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MtSpliceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["MtSpliceTower"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
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
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, MtSpliceSplineWeight):
            init.zeros_(module.kernel)


class MtSpliceModel(MtSplicePreTrainedModel):
    """
    The bare MTSplice tissue-specific backbone.

    MTSplice scores a cassette exon together with its flanking introns with two
    parallel dilated-convolution towers (an acceptor tower over the upstream
    region and a donor tower over the downstream region), positionally re-weights
    each tower with B-spline transformations, pools, and combines the two towers
    into a per-tissue delta-logit-PSI vector. The backbone returns the per-tissue
    score vector. For variant-effect prediction, pass both a reference and an
    alternative sequence; the backbone then also returns the per-tissue deltas.

    Examples:
        >>> import torch
        >>> from multimolecule import MtSpliceConfig, MtSpliceModel
        >>> config = MtSpliceConfig()
        >>> model = MtSpliceModel(config)
        >>> _ = model.eval()
        >>> input_ids = torch.randint(4, (1, 800))
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([1, 56])
    """

    def __init__(self, config: MtSpliceConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = MtSpliceEmbedding(config)
        self.acceptor_tower = MtSpliceTower(config, config.acceptor_length)
        self.donor_tower = MtSpliceTower(config, config.donor_length)
        self.pooler = MtSplicePooler()
        self.prediction = MtSplicePredictionHead(config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MtSpliceModelOutput | tuple[Tensor, ...]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        reference = self._score(input_ids, attention_mask, inputs_embeds)

        delta = None
        alternative = None
        has_alternative = alternative_input_ids is not None or alternative_inputs_embeds is not None
        if has_alternative:
            if alternative_input_ids is not None and alternative_inputs_embeds is not None:
                raise ValueError("You cannot specify both alternative_input_ids and alternative_inputs_embeds")
            alternative = self._score(
                alternative_input_ids,
                alternative_attention_mask,
                alternative_inputs_embeds,
            )
            delta = alternative - reference

        return MtSpliceModelOutput(
            logits=reference,
            alternative_logits=alternative,
            delta_logits=delta,
        )

    def _score(
        self,
        input_ids: Tensor | NestedTensor | None,
        attention_mask: Tensor | None,
        inputs_embeds: Tensor | NestedTensor | None,
    ) -> Tensor:
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
        acceptor, donor = self._split(embedding_output)
        if self.gradient_checkpointing and self.training:
            acceptor = self._gradient_checkpointing_func(self.acceptor_tower.__call__, acceptor)
            donor = self._gradient_checkpointing_func(self.donor_tower.__call__, donor)
        else:
            acceptor = self.acceptor_tower(acceptor)
            donor = self.donor_tower(donor)
        pooled = self.pooler(acceptor, donor)
        return self.prediction(pooled)

    def _split(self, inputs_embeds: Tensor) -> tuple[Tensor, Tensor]:
        length = inputs_embeds.size(-1)
        acceptor_length = min(self.config.acceptor_length, length)
        donor_length = min(self.config.donor_length, length)
        acceptor = inputs_embeds[..., :acceptor_length]
        donor = inputs_embeds[..., length - donor_length :]
        return acceptor, donor


class MtSpliceForSequencePrediction(MtSplicePreTrainedModel):
    """
    MTSplice with sequence-level regression loss support.

    The wrapper returns the per-tissue score vector (or, when a reference and an
    alternative sequence are provided, the per-tissue score deltas) and applies a
    regression criterion when labels are supplied.

    Examples:
        >>> import torch
        >>> from multimolecule import MtSpliceConfig, MtSpliceForSequencePrediction
        >>> config = MtSpliceConfig()
        >>> model = MtSpliceForSequencePrediction(config)
        >>> _ = model.eval()
        >>> input_ids = torch.randint(4, (1, 800))
        >>> alternative_input_ids = torch.randint(4, (1, 800))
        >>> output = model(input_ids, alternative_input_ids=alternative_input_ids)
        >>> output["logits"].shape
        torch.Size([1, 56])
    """

    def __init__(self, config: MtSpliceConfig):
        super().__init__(config)
        self.model = MtSpliceModel(config)
        head = config.head
        if head is None:
            raise ValueError("MtSpliceForSequencePrediction requires `config.head` to be set")
        self.criterion = Criterion(head)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            alternative_input_ids=alternative_input_ids,
            alternative_attention_mask=alternative_attention_mask,
            alternative_inputs_embeds=alternative_inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        features = outputs.delta_logits if outputs.delta_logits is not None else outputs.logits
        loss = self.criterion(features, labels) if labels is not None else None
        return SequencePredictorOutput(loss=loss, logits=features)


class MtSpliceTower(nn.Module):
    """
    A single MTSplice sequence tower (acceptor or donor side).

    A stem convolution is followed by a stack of residual dilated-convolution
    blocks with exponentially growing receptive field, and a positional B-spline
    re-weighting of the resulting features.
    """

    def __init__(self, config: MtSpliceConfig, length: int):
        super().__init__()
        self.stem = nn.Conv1d(
            config.vocab_size,
            config.hidden_size,
            kernel_size=config.kernel_size,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]
        self.blocks = nn.ModuleList(
            [MtSpliceBlock(config, dilation=config.dilation_base ** (i + 1)) for i in range(config.num_blocks)]
        )
        self.spline = MtSpliceSplineWeight(config, length)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.act(self.stem(hidden_state))
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return self.spline(hidden_state)


class MtSpliceEmbedding(nn.Module):
    """
    One-hot encodes the input sequence using the MultiMolecule tokenizer order.

    Produces a ``(batch, vocab_size, length)`` tensor; the two towers slice their
    region along the length dimension.
    """

    def __init__(self, config: MtSpliceConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_tokens = self.vocab_size + 1
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
            inputs_embeds = F.one_hot(input_ids, num_classes=self.num_tokens)[..., : self.vocab_size].to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class MtSpliceBlock(nn.Module):
    """
    A residual dilated-convolution block.

    A batch norm precedes a dilated convolution; the convolution output is added
    to the running residual accumulator. The batch norm consumes the accumulator,
    so the spline layer receives the raw (un-normalized) accumulator.
    """

    def __init__(self, config: MtSpliceConfig, dilation: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.norm = nn.BatchNorm1d(hidden_size, eps=config.batch_norm_eps)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.block_kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act(self.conv(hidden_state))
        return residual + hidden_state


class MtSpliceSplineWeight(nn.Module):
    """
    Positional B-spline re-weighting of per-channel activations.

    ``x_out = x_in * (1 + B @ kernel)`` where ``B`` is a deterministic
    ``(length, spline_bases)`` cubic B-spline design matrix and ``kernel`` is a
    learned ``(spline_bases, channels)`` matrix.

    The design matrix is a deterministic constant that is *not* stored in the
    checkpoint: it is rebuilt from config inside ``forward`` (and cached per
    length/device/dtype). Keeping it out of the state dict avoids the
    transformers meta-init corruption of non-persistent buffers that would
    otherwise produce NaNs after ``from_pretrained``.
    """

    def __init__(self, config: MtSpliceConfig, length: int):
        super().__init__()
        self.length = length
        self.spline_bases = config.spline_bases
        self.spline_degree = config.spline_degree
        self.kernel = nn.Parameter(torch.zeros(config.spline_bases, config.hidden_size))
        self._basis_cache: dict[tuple, Tensor] = {}

    def _basis(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (length, device, dtype)
        basis = self._basis_cache.get(key)
        if basis is None:
            basis = torch.from_numpy(build_bspline_basis(length, self.spline_bases, self.spline_degree)).to(
                device=device, dtype=dtype
            )
            self._basis_cache[key] = basis
        return basis

    def forward(self, hidden_state: Tensor) -> Tensor:
        length = hidden_state.size(-1)
        basis = self._basis(length, hidden_state.device, hidden_state.dtype)
        spline = (basis @ self.kernel) + 1.0
        return hidden_state * spline.transpose(0, 1).unsqueeze(0)


class MtSplicePooler(nn.Module):
    """
    Concatenates the two towers along the length axis and average-pools.

    Mirrors the upstream ``Concatenate(axis=-2)`` followed by
    ``GlobalAveragePooling1D``; the pooled vector is the length-weighted mean of
    the two towers.
    """

    def forward(self, acceptor: Tensor, donor: Tensor) -> Tensor:
        hidden_state = torch.cat([acceptor, donor], dim=-1)
        return hidden_state.mean(dim=-1)


class MtSplicePredictionHead(nn.Module):
    def __init__(self, config: MtSpliceConfig):
        super().__init__()
        self.norm = nn.BatchNorm1d(config.hidden_size, eps=config.batch_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.mlp_size)
        self.act = ACT2FN[config.hidden_act]
        self.post_norm = nn.BatchNorm1d(config.mlp_size, eps=config.batch_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.decoder = nn.Linear(config.mlp_size, config.num_labels)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act(self.dense(hidden_state))
        hidden_state = self.post_norm(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return self.decoder(hidden_state)


def build_bspline_basis(length: int, n_bases: int, spline_degree: int) -> np.ndarray:
    """
    Construct the cubic B-spline design matrix used by the positional re-weighting.

    Reproduces the upstream ``BSpline`` knot construction (mgcv-style) and
    ``scipy.interpolate.splev`` evaluation so the deterministic constant matches
    the original Keras ``SplineWeight1D`` layer.
    """

    positions: np.ndarray = np.arange(length, dtype=np.float64)
    start = 0.0
    end = float(length - 1)
    x_range = end - start
    lo = start - x_range * 0.001
    hi = end + x_range * 0.001
    m = spline_degree - 1
    nk = n_bases - m
    dknots = (hi - lo) / (nk - 1)
    knots = np.linspace(start=lo - dknots * (m + 1), stop=hi + dknots * (m + 1), num=nk + 2 * m + 2)
    basis: np.ndarray = np.zeros((length, n_bases), dtype=np.float64)
    for i in range(n_bases):
        coeffs = np.zeros(n_bases)
        coeffs[i] = 1.0
        basis[:, i] = splev(positions, (knots, coeffs, spline_degree), der=0)
    return basis.astype(np.float32)


@dataclass
class MtSpliceModelOutput(ModelOutput):
    """
    Base class for outputs of the MTSplice tissue-specific model.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            The per-tissue delta-logit-PSI score vector for the (reference) input
            sequence, ordered as the 56 GTEx tissues (see `MtSpliceConfig`).
        alternative_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
            The per-tissue score vector for the alternative sequence, returned when
            an alternative sequence is provided.
        delta_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
            `alternative_logits - logits`, the per-tissue variant-effect deltas,
            returned when an alternative sequence is provided.
    """

    logits: torch.FloatTensor | None = None
    alternative_logits: torch.FloatTensor | None = None
    delta_logits: torch.FloatTensor | None = None
