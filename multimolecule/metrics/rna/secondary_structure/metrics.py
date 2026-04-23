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
import warnings
from collections.abc import Sequence
from typing import List

import torch
from danling import NestedTensor
from danling.metrics import AverageMeters
from torch import Tensor

from ....utils.rna.secondary_structure import LoopSegmentType, LoopType
from .common import (
    f1_from_confusion,
    pair_error_rate,
    pair_exact_match,
    precision_from_confusion,
    recall_from_confusion,
)
from .context import RnaSecondaryStructureContext
from .functional import binary_auprc, binary_auroc, binary_prf_mcc, f1_from_pr
from .graph import loop_helix_graph_ged
from .noncanonical import noncanonical_pairs_confusion
from .pseudoknot import crossing_pairs_confusion
from .stems import stem_confusion


class RnaSecondaryStructureMetrics(AverageMeters):

    threshold: float

    def __init__(self, *, threshold: float = 0.5) -> None:
        super().__init__()
        # Avoid chanfig meter coercion for plain attributes.
        self.setattr("threshold", float(threshold))

    def update(
        self,
        preds: Tensor | NestedTensor | Sequence[Tensor],
        targets: Tensor | NestedTensor | Sequence[Tensor],
        *,
        sequences: str | Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> None:
        if sequences is None:
            raise ValueError("sequences are required for RNA secondary structure metrics")
        pred_list = _as_tensor_list(preds, name="preds")
        target_list = _as_tensor_list(targets, name="targets")
        if len(pred_list) != len(target_list):
            raise ValueError("preds and targets must have the same batch size")
        seq_list = _as_sequence_list(sequences, expected=len(pred_list))
        if threshold is None:
            threshold = self.threshold
        did_warn_sigmoid = False

        def _update_meter(name: str, value: Tensor | float) -> None:
            if isinstance(value, Tensor):
                scalar = float(value.detach().item())
            else:
                scalar = float(value)
            if not math.isfinite(scalar):
                return
            self[name].update(scalar, n=1)

        def _update_confusion_family(prefix: str, confusion: Tensor, *, device: torch.device) -> None:
            _update_meter(f"{prefix}_precision", precision_from_confusion(confusion, device))
            _update_meter(f"{prefix}_recall", recall_from_confusion(confusion, device))
            _update_meter(f"{prefix}_f1", f1_from_confusion(confusion, device))

        def _update_binary_family(prefix: str, pred_labels: Tensor, target_labels: Tensor) -> None:
            precision, recall, f1, mcc = binary_prf_mcc(pred_labels, target_labels)
            _update_meter(f"{prefix}_precision", precision)
            _update_meter(f"{prefix}_recall", recall)
            _update_meter(f"{prefix}_f1", f1)
            _update_meter(f"{prefix}_mcc", mcc)

        def _update_overlap_family(prefix: str, precision: Tensor, recall: Tensor) -> None:
            _update_meter(f"{prefix}_precision", precision)
            _update_meter(f"{prefix}_recall", recall)
            _update_meter(f"{prefix}_f1", f1_from_pr(precision, recall))

        for pred_map, target_map, sequence in zip(pred_list, target_list, seq_list):
            n = target_map.shape[0]
            tri = torch.triu_indices(n, n, offset=1, device=target_map.device)
            pred_lin = pred_map[tri[0], tri[1]]
            target_lin = target_map[tri[0], tri[1]].to(dtype=torch.int64)
            if pred_lin.is_floating_point():
                pred_scores = pred_lin.to(dtype=torch.float32)
                if pred_scores.numel():
                    min_val = float(pred_scores.min().item())
                    max_val = float(pred_scores.max().item())
                    if min_val < 0.0 or max_val > 1.0:
                        if not did_warn_sigmoid:
                            warnings.warn(
                                "Applying sigmoid to predictions outside [0, 1].",
                                UserWarning,
                                stacklevel=2,
                            )
                            did_warn_sigmoid = True
                        pred_scores = pred_scores.sigmoid()
                pred_labels = pred_scores >= threshold
            else:
                pred_scores = pred_lin.to(dtype=torch.float32)
                pred_labels = pred_lin
            pred_labels = pred_labels.to(dtype=torch.int64)
            binary_precision, binary_recall, binary_f1, binary_mcc = binary_prf_mcc(pred_labels, target_lin)
            _update_meter("binary_precision", binary_precision)
            _update_meter("binary_recall", binary_recall)
            _update_meter("binary_f1", binary_f1)
            _update_meter("binary_mcc", binary_mcc)
            _update_meter("binary_auroc", binary_auroc(pred_scores, target_lin))
            _update_meter("binary_auprc", binary_auprc(pred_scores, target_lin))
            context = RnaSecondaryStructureContext(pred_map, target_map, sequence, threshold=threshold)
            pred_pairs = context.pred_pairs
            target_pairs = context.target_pairs
            _update_meter("pair_exact_match", pair_exact_match(pred_pairs, target_pairs, context.device))
            _update_meter("pair_error_rate", pair_error_rate(pred_pairs, target_pairs, context.device))
            paired_preds, paired_targets = context.paired_labels
            _update_binary_family("paired_nucleotides", paired_preds, paired_targets)

            loop_families = (
                ("hairpin", LoopType.HAIRPIN, LoopSegmentType.HAIRPIN),
                ("bulge", LoopType.BULGE, LoopSegmentType.BULGE),
                ("internal", LoopType.INTERNAL, LoopSegmentType.INTERNAL),
                ("multiloop", LoopType.MULTILOOP, LoopSegmentType.BRANCH),
                ("external", LoopType.EXTERNAL, LoopSegmentType.EXTERNAL),
            )
            for loop_prefix, loop_type, segment_type in loop_families:
                loop_precision, loop_recall = context.loop_overlap_ratios(loop_type)
                _update_overlap_family(f"{loop_prefix}_loops", loop_precision, loop_recall)
                nt_preds, nt_targets = context.loop_nt_labels(loop_type)
                _update_binary_family(f"{loop_prefix}_nucleotides", nt_preds, nt_targets)
                segment_precision, segment_recall = context.segment_overlap_ratios(segment_type)
                _update_overlap_family(f"{loop_prefix}_segments", segment_precision, segment_recall)

            end_precision, end_recall = context.segment_overlap_ratios(LoopSegmentType.END)
            _update_overlap_family("end_segments", end_precision, end_recall)

            _update_confusion_family("loop_helix_edges", context.loop_helix_edges_confusion, device=context.device)
            _update_meter("loop_helix_graph_ged", loop_helix_graph_ged(context))
            _update_confusion_family("topology", context.topology_confusion, device=context.device)
            _update_meter("topology_ged", context.topology_ged)

            stem_topology_confusion = stem_confusion(context)
            _update_confusion_family("stem", stem_topology_confusion, device=context.device)
            _update_confusion_family("stem_pairs", context.stem_confusion, device=context.device)
            _update_confusion_family("crossing_stem", context.crossing_stem_confusion, device=context.device)
            _update_confusion_family("noncanonical_pairs", noncanonical_pairs_confusion(context), device=context.device)
            _update_confusion_family("crossing_pairs", crossing_pairs_confusion(context), device=context.device)

            crossing_preds, crossing_targets = context.crossing_nt_labels
            _update_binary_family("crossing_nucleotides", crossing_preds, crossing_targets)
            _update_confusion_family("crossing_events", context.crossing_events_confusion, device=context.device)


def _as_tensor_list(values: Tensor | NestedTensor | Sequence[Tensor], *, name: str) -> List[Tensor]:
    if isinstance(values, NestedTensor):
        return list(values)
    if isinstance(values, Tensor):
        if values.ndim == 2:
            return [values]
        if values.ndim == 3:
            return list(values)
        if values.ndim == 4 and values.shape[-1] == 1:
            return [item.squeeze(-1) for item in values]
        raise ValueError(f"{name} must be a 2D, 3D, or (B, L, L, 1) tensor")
    if isinstance(values, Sequence):
        tensors: list[Tensor] = []
        for item in values:
            if isinstance(item, Tensor):
                tensors.append(item)
            else:
                tensors.append(torch.as_tensor(item))
        return tensors
    raise TypeError(f"{name} must be a Tensor, NestedTensor, or sequence of tensors")


def _as_sequence_list(sequences: str | Sequence[str], *, expected: int) -> List[str]:
    if isinstance(sequences, str):
        seq_list = [sequences]
    elif isinstance(sequences, Sequence):
        seq_list = [str(seq) for seq in sequences]
    else:
        raise TypeError("sequences must be a string or sequence of strings")
    if len(seq_list) != expected:
        raise ValueError("number of sequences must match batch size")
    return seq_list
