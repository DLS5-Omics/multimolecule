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

from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers import initialization as init
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import Criterion

from ..modeling_outputs import SequencePredictorOutput
from .configuration_hal import HalConfig

# MultiMolecule streamline RNA tokenizer order is ["A", "C", "G", "U", "N", ...].
# Only the four canonical nucleotides contribute hexamer features; the index map below assigns
# each canonical token id a base-`nucleobase_size` digit and marks every other id as -1 (ignored).
CANONICAL_TOKENS = ("A", "C", "G", "U")


class HalPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HalConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["HalModule"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # The HAL hexamer-coefficient layer is the actual published model weight. It is
        # zero-initialized here so a freshly constructed model is well-defined before the
        # converter loads the published coefficient table.
        if isinstance(module, nn.Linear):
            init.zeros_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)


class HalModel(HalPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HalConfig, HalModel
        >>> config = HalConfig()
        >>> model = HalModel(config)
        >>> output = model(torch.randint(4, (1, config.region_length)))
        >>> output["pooler_output"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: HalConfig):
        super().__init__(config)
        self.embeddings = HalEmbedding(config)
        self.prediction = HalModule(config)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def dtype(self) -> torch.dtype:
        """Active dtype of the HAL coefficient matrix."""
        return self.prediction.prediction.weight.dtype

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Any,
    ) -> HalModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            inputs_embeds = inputs_embeds.tensor

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embeddings(input_ids, attention_mask=attention_mask, dtype=self.dtype)
        else:
            if inputs_embeds.dim() == 1:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            inputs_embeds = inputs_embeds.to(dtype=self.dtype)

        score = self.prediction(inputs_embeds)

        return HalModelOutput(pooler_output=score, hexamer_frequencies=inputs_embeds)


class HalForSequencePrediction(HalPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import HalConfig, HalForSequencePrediction
        >>> config = HalConfig()
        >>> model = HalForSequencePrediction(config)
        >>> output = model(torch.randint(4, (1, config.region_length)), labels=torch.tensor([[1.0]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: HalConfig):
        super().__init__(config)
        self.model = HalModel(config)
        head = config.head
        if head is None:
            raise ValueError("HalForSequencePrediction requires `config.head` to be set")
        self.criterion = Criterion(head)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.pooler_output
        if logits is None:
            raise RuntimeError("HalModel did not return `pooler_output`")
        loss = self.criterion(logits, labels) if labels is not None else None
        return SequencePredictorOutput(loss=loss, logits=logits)


class HalModule(nn.Module):
    r"""
    The HAL linear model over normalized hexamer-frequency features.

    The hexamer coefficient matrix is the actual published HAL model weight, so it is stored as a persistent
    [`nn.Linear`] parameter rather than a deterministic constant.
    """

    def __init__(self, config: HalConfig):
        super().__init__()
        self.prediction = nn.Linear(config.num_features, config.num_labels, bias=False)

    def forward(self, hexamer_frequencies: Tensor) -> Tensor:
        return self.prediction(hexamer_frequencies)


class HalEmbedding(nn.Module):
    r"""
    Converts MultiMolecule token ids into normalized hexamer (k-mer) frequency features.

    The k-mer index map is a deterministic constant fully determined by the configuration, so it is *not*
    a checkpoint tensor. It is rebuilt lazily in `forward` from config rather than relied upon as a loaded
    buffer: under Transformers v5 `from_pretrained` materializes modules on the meta device and only restores
    tensors present in the checkpoint, so a non-persistent buffer set in `__init__` would be left uninitialized.
    """

    def __init__(self, config: HalConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.kmer_size = config.kmer_size
        self.nucleobase_size = config.nucleobase_size
        self.num_kmers = config.num_kmers
        self.region_length = config.region_length

    def token_to_digit(self, device: torch.device) -> Tensor:
        # Deterministic index map rebuilt from config: each vocabulary id maps to its
        # base-`nucleobase_size` digit, or -1 if the token is not a canonical nucleotide.
        token_to_digit = torch.full((self.vocab_size,), -1, dtype=torch.long, device=device)
        for token, digit in zip(CANONICAL_TOKENS[: self.nucleobase_size], range(self.nucleobase_size)):
            token_to_digit[CANONICAL_TOKENS.index(token)] = digit
        return token_to_digit

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any()):
            raise ValueError(f"HAL token ids must be in [0, {self.vocab_size}), got values outside that range")
        digits = self.token_to_digit(input_ids.device)[input_ids]
        valid = digits >= 0
        if attention_mask is not None:
            valid = valid & attention_mask.bool()

        if self.region_length and input_ids.size(1) != self.region_length:
            raise ValueError(
                f"Expected sequence region of length {self.region_length}, but got {input_ids.size(1)}. "
                "HAL follows the published fixed 160-nucleotide scoring window."
            )
        batch_size, seq_length = input_ids.shape
        num_windows = seq_length - self.kmer_size + 1
        frequencies = torch.zeros(batch_size, self.num_kmers, device=input_ids.device, dtype=dtype)
        if num_windows <= 0:
            return frequencies

        powers = self.nucleobase_size ** torch.arange(
            self.kmer_size - 1, -1, -1, device=input_ids.device, dtype=torch.long
        )
        for start in range(num_windows):
            window_digits = digits[:, start : start + self.kmer_size]
            window_valid = valid[:, start : start + self.kmer_size].all(dim=1)
            # No early-continue guard here: scatter_add_ with all-zero `update` is a no-op,
            # so skipping the sync avoids a GPU→CPU round-trip on every iteration.
            indices = (window_digits.clamp(min=0) * powers).sum(dim=1)
            indices = torch.where(window_valid, indices, torch.zeros_like(indices))
            update = window_valid.to(dtype)
            frequencies.scatter_add_(1, indices.unsqueeze(1), update.unsqueeze(1))

        totals = frequencies.sum(dim=1, keepdim=True).clamp(min=1.0)
        return frequencies / totals


@dataclass
class HalModelOutput(ModelOutput):
    """
    Base class for outputs of the HAL model.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            The HAL splicing score predicted by the linear hexamer model.
        hexamer_frequencies (`torch.FloatTensor` of shape `(batch_size, num_kmers)`, *optional*):
            The normalized hexamer (k-mer) frequency features derived from the input sequence region.
        hidden_states:
            Always `None`; HAL is a single linear layer and has no intermediate hidden states. Provided for
            compatibility with the Transformers output convention.
        attentions:
            Always `None`; HAL has no attention layers. Provided for compatibility with the Transformers output
            convention.
    """

    pooler_output: Tensor | None = None
    hexamer_frequencies: Tensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
