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

import importlib
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.utils import pairs_to_contact_map

from .configuration_mxfold2 import Mxfold2Config

TOKEN_TO_BASE = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
}


@dataclass(frozen=True)
class _Mxfold2Backend:
    MixedFold: type[nn.Module]
    RNAFold: type[nn.Module]
    ZukerFold: type[nn.Module]
    param_turner2004: Any


class Mxfold2PreTrainedModel(PreTrainedModel):

    config_class = Mxfold2Config
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None


class Mxfold2Model(Mxfold2PreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import Mxfold2Config, Mxfold2Model
        >>> config = Mxfold2Config()
        >>> model = Mxfold2Model(config)
        >>> input_ids = torch.tensor([[3, 0, 2, 1, 3, 3]])
        >>> output = model(input_ids=input_ids)
        >>> output["contact_map"].shape
        torch.Size([1, 6, 6])
    """

    def __init__(self, config: Mxfold2Config):
        super().__init__(config)
        self.model = _build_backend_model(config)
        self.supports_batch_process = False
        self._refresh_length_layer_basis()
        self.post_init()

    def postprocess(self, outputs, input_ids=None, **kwargs):
        return outputs["contact_map"]

    def get_input_embeddings(self):
        embedding = getattr(getattr(getattr(self.model, "zuker", None), "net", None), "embedding", None)
        return getattr(embedding, "embedding", None)

    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        constraint: Tensor | None = None,
        reference: Tensor | None = None,
        return_partfunc: bool = False,
        max_internal_length: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> "Mxfold2ModelOutput":
        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        if inputs_embeds is not None:
            raise TypeError("Mxfold2Model does not support inputs_embeds because MXfold2 decodes from sequence tokens.")
        if labels is not None:
            raise NotImplementedError("Mxfold2Model does not expose training losses in the MM wrapper.")
        if input_ids is None:
            raise ValueError("You have to specify input_ids.")

        self._refresh_length_layer_basis()
        sequences = _decode_sequences(input_ids, attention_mask)
        max_internal_length = self.config.max_internal_length if max_internal_length is None else max_internal_length

        with torch.no_grad():
            outputs = self.model(
                sequences,
                return_partfunc=return_partfunc,
                max_internal_length=max_internal_length,
                constraint=constraint,
                reference=reference,
            )
        if return_partfunc:
            scores, secondary_structure, pair_tables, _, base_pair_probs = outputs
        else:
            scores, secondary_structure, pair_tables = outputs
            base_pair_probs = None

        device = input_ids.device
        lengths = [len(sequence) for sequence in sequences]
        max_length = max(lengths, default=0)
        contact_map = torch.zeros((len(sequences), max_length, max_length), dtype=torch.float32, device=device)
        pair_tensors = []

        for batch_idx, (pair_table, length) in enumerate(zip(pair_tables, lengths)):
            pairs = _bpseq_to_pairs_tensor(pair_table, device=device)
            pair_tensors.append(pairs)
            if pairs.numel() == 0:
                continue
            predicted = pairs_to_contact_map(pairs, length=length).to(dtype=torch.float32, device=device)
            contact_map[batch_idx, :length, :length] = predicted

        probs = None
        if base_pair_probs is not None:
            probs = torch.zeros((len(sequences), max_length, max_length), dtype=torch.float32, device=device)
            for batch_idx, (bpp, length) in enumerate(zip(base_pair_probs, lengths)):
                probs[batch_idx, :length, :length] = torch.as_tensor(bpp[1:, 1:], dtype=torch.float32, device=device)

        return Mxfold2ModelOutput(
            logits=contact_map,
            scores=scores.to(device),
            contact_map=contact_map,
            secondary_structure=tuple(secondary_structure),
            pairs=tuple(pair_tensors),
            base_pair_probs=probs,
        )

    def _refresh_length_layer_basis(self) -> None:
        for module in self.model.modules():
            if module.__class__.__name__ != "LengthLayer" or not hasattr(module, "n_in"):
                continue
            basis = getattr(module, "x", None)
            if basis is not None and not getattr(basis, "is_meta", False):
                continue
            module.x = _build_length_basis(module.n_in)


def _build_backend_model(config: Mxfold2Config) -> nn.Module:
    backend = _load_backend()
    kwargs = {
        "max_helix_length": config.max_helix_length,
        "embed_size": config.embed_size,
        "num_filters": tuple(config.num_filters),
        "filter_size": tuple(config.filter_size),
        "pool_size": tuple(config.pool_size),
        "dilation": config.dilation,
        "num_lstm_layers": config.num_lstm_layers,
        "num_lstm_units": config.num_lstm_units,
        "num_transformer_layers": config.num_transformer_layers,
        "num_transformer_hidden_units": config.num_transformer_hidden_units,
        "num_transformer_att": config.num_transformer_att,
        "num_hidden_units": tuple(config.num_hidden_units),
        "num_paired_filters": tuple(config.num_paired_filters),
        "paired_filter_size": tuple(config.paired_filter_size),
        "dropout_rate": config.dropout_rate,
        "fc_dropout_rate": config.fc_dropout_rate,
        "num_att": config.num_att,
        "pair_join": config.pair_join,
        "no_split_lr": config.no_split_lr,
    }

    if config.folding_model == "Turner":
        return backend.RNAFold(backend.param_turner2004)
    if config.folding_model == "Zuker":
        return backend.ZukerFold(model_type="M", **kwargs)
    if config.folding_model == "ZukerS":
        return backend.ZukerFold(model_type="S", **kwargs)
    if config.folding_model == "ZukerL":
        return backend.ZukerFold(model_type="L", **kwargs)
    if config.folding_model == "ZukerC":
        return backend.ZukerFold(model_type="C", **kwargs)
    if config.folding_model == "Mix":
        return backend.MixedFold(init_param=backend.param_turner2004, model_type="M", **kwargs)
    if config.folding_model == "MixC":
        return backend.MixedFold(init_param=backend.param_turner2004, model_type="C", **kwargs)
    raise ValueError(f"Unsupported MXfold2 folding model: {config.folding_model}.")


@lru_cache(maxsize=1)
def _load_backend() -> _Mxfold2Backend:
    try:
        return _import_backend()
    except Exception as first_error:
        backend_root = _repo_backend_root()
        if backend_root is None:
            raise ImportError("Mxfold2Model requires the `mxfold2` backend package.") from first_error
        _prepare_repo_backend(backend_root)
        try:
            return _import_backend()
        except Exception as second_error:
            raise ImportError(
                "Unable to import the MXfold2 backend from the installed environment or the local `mxfold2-code` tree."
            ) from second_error


def _import_backend() -> _Mxfold2Backend:
    mix_module = importlib.import_module("mxfold2.fold.mix")
    rnafold_module = importlib.import_module("mxfold2.fold.rnafold")
    zuker_module = importlib.import_module("mxfold2.fold.zuker")
    turner_module = importlib.import_module("mxfold2.param_turner2004")
    return _Mxfold2Backend(
        MixedFold=mix_module.MixedFold,
        RNAFold=rnafold_module.RNAFold,
        ZukerFold=zuker_module.ZukerFold,
        param_turner2004=turner_module,
    )


def _repo_backend_root() -> Path | None:
    backend_root = Path(__file__).resolve().parents[3] / "mxfold2-code"
    return backend_root if backend_root.exists() else None


def _prepare_repo_backend(backend_root: Path) -> None:
    backend_path = str(backend_root)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    package_dir = backend_root / "mxfold2"
    if any(package_dir.glob("interface*.so")):
        importlib.invalidate_caches()
        return
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    process = subprocess.run(command, cwd=backend_root, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            "Failed to build the local MXfold2 backend.\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )
    importlib.invalidate_caches()


def _decode_sequences(input_ids: Tensor, attention_mask: Tensor | None) -> list[str]:
    input_ids = input_ids.detach().cpu()
    attention_mask = attention_mask.detach().cpu() if attention_mask is not None else None
    sequences = []
    for batch_idx, row in enumerate(input_ids.tolist()):
        if attention_mask is None:
            tokens = row
        else:
            tokens = [token for token, keep in zip(row, attention_mask[batch_idx].tolist()) if keep]
        sequences.append("".join(TOKEN_TO_BASE.get(token, "N") for token in tokens))
    return sequences


def _bpseq_to_pairs_tensor(pair_table: list[int], device: torch.device) -> Tensor:
    pairs = [(index - 1, partner - 1) for index, partner in enumerate(pair_table[1:], start=1) if partner > index]
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _build_length_basis(n_in: int | tuple[int, int]) -> Tensor:
    if isinstance(n_in, int):
        return torch.tril(torch.ones((n_in, n_in)))
    n = int(np.prod(n_in))
    basis = np.fromfunction(lambda i, j, k, l: np.logical_and(k <= i, l <= j), (*n_in, *n_in))
    return torch.from_numpy(basis.astype(np.float32)).reshape(n, n)


@dataclass
class Mxfold2ModelOutput(ModelOutput):
    """
    Output type for MXfold2.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    scores: torch.FloatTensor | None = None
    contact_map: torch.FloatTensor | None = None
    secondary_structure: tuple[str, ...] | None = None
    pairs: tuple[torch.LongTensor, ...] | None = None
    base_pair_probs: torch.FloatTensor | None = None
