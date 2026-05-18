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

import os
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import Criterion

from ..modeling_outputs import SequencePredictorOutput
from .configuration_maxentscan import MaxEntScanConfig

# The streamline RNA alphabet is `ACGUN`. MaxEntScan only scores `ACGU`; the maximum-entropy
# tables are indexed in base-4 with the order A=0, C=1, G=2, U=3, which matches the streamline
# token order, so token ids are used directly as base-4 digits. This is exactly the
# original `tr/ACGT/0123/` mapping with upstream T represented as RNA U.

# score5: the 9-mer is `(exon)XXX|XXXXXX(intron)`; the consensus dinucleotide (GT) sits at
# positions 3 and 4. The remaining 7 positions (0,1,2,5,6,7,8) form the `rest` 7-mer that
# indexes the 16384-entry `me2x5` maximum-entropy table (the published `splice5sequences`
# enumeration is exactly base-4 order, so the base-4 hash of the 7-mer is the table index).
SCORE5_REST_POSITIONS: tuple[int, ...] = (0, 1, 2, 5, 6, 7, 8)
SCORE5_CONS_POSITIONS: tuple[int, int] = (3, 4)

# score3: the 23-mer is `(intron)X*20|XXX(exon)`; the consensus dinucleotide (AG) sits at
# positions 18 and 19. Dropping it leaves the 21-mer `rest` = positions 0..17 + 20,21,22.
# The 21-mer is scored by the maximum-entropy decomposition of nine overlapping submodels
# (Yeo & Burge 2004); `rest` slices below are 0-indexed into the 21-mer.
SCORE3_CONS_POSITIONS: tuple[int, int] = (18, 19)
SCORE3_REST_POSITIONS: tuple[int, ...] = tuple(range(0, 18)) + (20, 21, 22)
# (start, stop) half-open slices into the 21-mer `rest`, matching `&maxentscore` in score3.pl.
SCORE3_SUBMODEL_POSITIONS: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 7)),
    tuple(range(7, 14)),
    tuple(range(14, 21)),
    tuple(range(4, 11)),
    tuple(range(11, 18)),
    tuple(range(4, 7)),
    tuple(range(7, 11)),
    tuple(range(11, 14)),
    tuple(range(14, 18)),
)
# The decomposition score is prod(numerator submodels) / prod(denominator submodels).
SCORE3_NUMERATOR = (0, 1, 2, 3, 4)
SCORE3_DENOMINATOR = (5, 6, 7, 8)

# The published Yeo & Burge (2004) maximum-entropy probability tables are bundled, verbatim, as
# plain-text files alongside this module (one float per line, native = base-4 order). They are the
# fixed "parameters" of MaxEntScan and are loaded into the score-table buffers on every `__init__`,
# so the model is correct without any external checkpoint. The score tables are persistent buffers
# (they are the model's parameters and serialize into the saved checkpoint); the derived
# constants below stay non-persistent and are reconstructed in `__init__`.
_TABLE_DIR = os.path.dirname(__file__)
SCORE5_TABLE_FILE = os.path.join(_TABLE_DIR, "score5_me2x5.txt")
SCORE3_TABLE_FILE = os.path.join(_TABLE_DIR, "score3_me2x3acc.txt")


class MaxEntScanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle the fixed maximum-entropy score tables and a simple interface for downloading and
    loading the published MaxEntScan parameters.
    """

    config_class = MaxEntScanConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None

    @torch.no_grad()
    def _init_weights(self, module):
        # MaxEntScan has no trainable parameters; nothing to initialize.
        return

    @property
    def dtype(self) -> torch.dtype:
        # MaxEntScan has no `nn.Parameter`; the base `PreTrainedModel.dtype` iterates
        # `self.parameters()` and raises `StopIteration` for a parameter-free model. Fall back
        # to the dtype of the first floating-point buffer (or float32 if none is set yet).
        for tensor in self.buffers():
            if tensor.is_floating_point():
                return tensor.dtype
        return torch.float32

    @property
    def device(self) -> torch.device:
        # MaxEntScan has no `nn.Parameter`; the base `PreTrainedModel.device` iterates
        # `self.parameters()` and raises `StopIteration` for a parameter-free model. Fall back
        # to the device of the first score-table buffer.
        for tensor in self.buffers():
            return tensor.device
        return torch.device("cpu")


class MaxEntScanModel(MaxEntScanPreTrainedModel):
    """
    Maximum-entropy splice-site scorer (Yeo & Burge, 2004).

    The model has no trainable weights. It exposes a single maximum-entropy score per input window through fixed
    score-table buffers populated from the published Yeo & Burge (2004) tables.

    Examples:
        >>> import torch
        >>> from multimolecule import MaxEntScanConfig, MaxEntScanModel
        >>> config = MaxEntScanConfig()
        >>> model = MaxEntScanModel(config)
        >>> output = model(torch.randint(4, (1, config.window)))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: MaxEntScanConfig):
        super().__init__(config)
        self.mode = config.mode
        self.window = config.window
        self.scorer = MaxEntScanScorer(config)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Any,
    ) -> SequencePredictorOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is not None:
            raise ValueError("MaxEntScan scores discrete token windows and does not support inputs_embeds")
        assert input_ids is not None  # narrowed: both-None and inputs_embeds-not-None are rejected above
        if isinstance(input_ids, NestedTensor):
            input_ids = input_ids.tensor
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.size(1) != self.window:
            raise ValueError(
                f"MaxEntScan {self.mode} expects a fixed window of {self.window} tokens, " f"got {input_ids.size(1)}"
            )
        score = self.scorer(input_ids)
        # The maximum-entropy score is exposed through `logits`; the downstream head reads it via `output_name`.
        return SequencePredictorOutput(logits=score)


class MaxEntScanForSequencePrediction(MaxEntScanPreTrainedModel):
    """
    MaxEntScan scorer with sequence-level regression loss support.

    Examples:
        >>> import torch
        >>> from multimolecule import MaxEntScanConfig, MaxEntScanForSequencePrediction
        >>> config = MaxEntScanConfig()
        >>> model = MaxEntScanForSequencePrediction(config)
        >>> output = model(torch.randint(4, (1, config.window)), labels=torch.randn(1, 1))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: MaxEntScanConfig):
        super().__init__(config)
        self.model = MaxEntScanModel(config)
        head = config.head
        if head is None:
            raise ValueError("MaxEntScanForSequencePrediction requires `config.head` to be set")
        # MaxEntScan is parameter-free: the score is passed straight to `Criterion` with no trainable head.
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
        logits = outputs.logits
        loss = self.criterion(logits, labels) if labels is not None else None
        return SequencePredictorOutput(loss=loss, logits=logits)


class MaxEntScanScorer(nn.Module):
    """
    Computes the maximum-entropy splice-site score from fixed score-table buffers.

    The implementation mirrors the original Yeo & Burge (2004) `score5.pl` / `score3.pl` exactly:

    ``score = log2( consensus_ratio * rest_score )``

    where ``consensus_ratio = cons1[b0] * cons2[b1] / (bgd[b0] * bgd[b1])`` over the two consensus
    positions, and ``rest_score`` is:

    * ``score5``: the published 16384-entry ``me2x5`` maximum-entropy probability indexed by the
      base-4 hash of the 7 non-consensus positions.
    * ``score3``: the maximum-entropy decomposition ``sc0*sc1*sc2*sc3*sc4 / (sc5*sc6*sc7*sc8)``
      over the nine overlapping ``me2x3acc1..9`` submodels of the 21-mer non-consensus sequence.
    """

    # Background nucleotide frequencies (identical for both modes) and the hardcoded consensus
    # probabilities at the two consensus positions, order A,C,G,T (from `score5.pl`/`score3.pl`).
    BGD = (0.27, 0.23, 0.23, 0.27)
    CONS = {
        "score5": ((0.004, 0.0032, 0.9896, 0.0032), (0.0034, 0.0039, 0.0042, 0.9884)),
        "score3": ((0.9903, 0.0032, 0.0034, 0.0030), (0.0027, 0.0037, 0.9905, 0.0030)),
    }
    bgd: Tensor
    score5_cons1: Tensor
    score5_cons2: Tensor
    score5_me2x5: Tensor
    score3_cons1: Tensor
    score3_cons2: Tensor
    _dtype_reference: Tensor

    def __init__(self, config: MaxEntScanConfig):
        super().__init__()
        self.mode = config.mode
        self.window = config.window
        # The published tables ARE the model's parameters (there is no upstream checkpoint), so
        # they are persistent and serialize into the saved checkpoint.
        if self.mode == "score5":
            self.register_buffer("score5_me2x5", load_score5_table(), persistent=True)
        else:
            for index, table in enumerate(load_score3_tables()):
                self.register_buffer(f"score3_table_{index}", table, persistent=True)
        # The small derived constants (background / consensus ratios) and the dtype tracker are
        # non-persistent: they are reconstructed lazily on the first `forward`. This mirrors the
        # transformers-v5 meta-init workaround used by other table-based MultiMolecule models
        # (`from_pretrained` does not restore non-persistent buffers, leaving them uninitialised).
        self._constants_ready = False
        self.register_buffer("bgd", torch.tensor(self.BGD), persistent=False)
        cons1, cons2 = self.CONS[self.mode]
        self.register_buffer(f"{self.mode}_cons1", torch.tensor(cons1), persistent=False)
        self.register_buffer(f"{self.mode}_cons2", torch.tensor(cons2), persistent=False)
        # Zero-size buffer that tracks the model's active dtype so the score is cast consistently
        # after .half() / .to(bf16).
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def _ensure_constants(self, device: torch.device) -> None:
        # Rebuild the non-persistent constant buffers on the input device. After `from_pretrained`
        # these buffers hold uninitialised memory, so they must be restored before first use.
        if self._constants_ready and self.bgd.device == device:
            return
        self.register_buffer("bgd", torch.tensor(self.BGD, device=device), persistent=False)
        cons1, cons2 = self.CONS[self.mode]
        self.register_buffer(f"{self.mode}_cons1", torch.tensor(cons1, device=device), persistent=False)
        self.register_buffer(f"{self.mode}_cons2", torch.tensor(cons2, device=device), persistent=False)
        if self._dtype_reference.device != device:
            self.register_buffer(
                "_dtype_reference", torch.empty(0, dtype=self._dtype_reference.dtype, device=device), persistent=False
            )
        self._constants_ready = True

    def forward(self, input_ids: Tensor) -> Tensor:
        self._ensure_constants(input_ids.device)
        dtype = self._dtype_reference.dtype if self._dtype_reference.dtype.is_floating_point else torch.float32
        # MaxEntScan only models ACGU; clamp the `N`/unknown token onto A so lookups stay in range.
        bases = input_ids.clamp(max=3).long()
        if self.mode == "score5":
            score = self._score5(bases)
        else:
            score = self._score3(bases)
        return score.to(dtype=dtype).unsqueeze(-1)

    def _hash(self, bases: Tensor, positions: tuple[int, ...]) -> Tensor:
        # Base-4 hash, most-significant digit first, matching `&hashseq` with T represented as U.
        index = torch.zeros(bases.size(0), dtype=torch.long, device=bases.device)
        for pos in positions:
            index = index * 4 + bases[:, pos]
        return index

    def _consensus(self, bases: Tensor, positions: tuple[int, int], cons1: Tensor, cons2: Tensor) -> Tensor:
        p0, p1 = positions
        b0, b1 = bases[:, p0], bases[:, p1]
        return cons1[b0] * cons2[b1] / (self.bgd[b0] * self.bgd[b1])

    def _score5(self, bases: Tensor) -> Tensor:
        consensus = self._consensus(bases, SCORE5_CONS_POSITIONS, self.score5_cons1, self.score5_cons2)
        rest = self.score5_me2x5[self._hash(bases, SCORE5_REST_POSITIONS)]
        score = consensus * rest
        return torch.log2(score.clamp_min(torch.finfo(score.dtype).tiny))

    def _score3(self, bases: Tensor) -> Tensor:
        consensus = self._consensus(bases, SCORE3_CONS_POSITIONS, self.score3_cons1, self.score3_cons2)
        # `rest` is the 21-mer with the AG consensus removed; submodels slice into this 21-mer.
        rest = bases[:, SCORE3_REST_POSITIONS]
        decomposition = torch.ones(bases.size(0), dtype=consensus.dtype, device=bases.device)
        for index, positions in enumerate(SCORE3_SUBMODEL_POSITIONS):
            table = getattr(self, f"score3_table_{index}")
            value = table[self._hash(rest, positions)]
            if index in SCORE3_NUMERATOR:
                decomposition = decomposition * value
            else:  # index in SCORE3_DENOMINATOR
                decomposition = decomposition / value
        score = consensus * decomposition
        return torch.log2(score.clamp_min(torch.finfo(score.dtype).tiny))


# ---------------------------------------------------------------------------
# Module-level table-loading helpers (placed after classes per style guide)
# ---------------------------------------------------------------------------


def _read_floats(path: str) -> list[float]:
    with open(path) as file:
        return [float(line) for line in file if line.strip() and not line.lstrip().startswith("#")]


def load_score5_table() -> Tensor:
    """Load the published 16384-entry ``me2x5`` 5'ss maximum-entropy probability table."""
    values = _read_floats(SCORE5_TABLE_FILE)
    if len(values) != 4**7:
        raise ValueError(f"{SCORE5_TABLE_FILE}: expected {4 ** 7} probabilities, found {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def load_score3_tables() -> list[Tensor]:
    """Load the published ``me2x3acc1..9`` 3'ss maximum-entropy decomposition tables."""
    sizes = [4 ** len(positions) for positions in SCORE3_SUBMODEL_POSITIONS]
    values = _read_floats(SCORE3_TABLE_FILE)
    if len(values) != sum(sizes):
        raise ValueError(f"{SCORE3_TABLE_FILE}: expected {sum(sizes)} probabilities, found {len(values)}")
    tables, offset = [], 0
    for size in sizes:
        tables.append(torch.tensor(values[offset : offset + size], dtype=torch.float32))
        offset += size
    return tables
