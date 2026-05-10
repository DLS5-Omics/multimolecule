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

from typing import TypedDict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from multimolecule.utils.rna.secondary_structure.notations import dot_bracket_to_pairs

from ..palettes import PAIR_COMPARISON_COLORS


class SecondaryStructureComparison(TypedDict):
    true_positive_pairs: list[tuple[int, int]]
    false_positive_pairs: list[tuple[int, int]]
    false_negative_pairs: list[tuple[int, int]]
    true_positives: int
    false_positives: int
    false_negatives: int


def compare_secondary_structures(
    predicted_dot_bracket: str,
    reference_dot_bracket: str,
) -> SecondaryStructureComparison:
    """
    Classify predicted and reference RNA secondary-structure pairs by base-pair identity.

    Args:
        predicted_dot_bracket: Predicted dot-bracket notation.
        reference_dot_bracket: Reference dot-bracket notation.

    Returns:
        A dictionary containing TP/FP/FN pair lists and counts. Pair-level metrics live in
        [multimolecule.metrics][]; this helper only returns visualization classes.

    Examples:
        >>> from multimolecule.visualization.rna import compare_secondary_structures
        >>> result = compare_secondary_structures("(((...)))", "((.....))")
        >>> result["true_positives"], result["false_positives"], result["false_negatives"]
        (2, 1, 0)
        >>> result["false_positive_pairs"]
        [(2, 6)]
    """
    if len(predicted_dot_bracket) != len(reference_dot_bracket):
        raise ValueError(
            "predicted_dot_bracket and reference_dot_bracket must have the same length "
            f"({len(predicted_dot_bracket)} != {len(reference_dot_bracket)})."
        )

    predicted_pairs = _pair_set(dot_bracket_to_pairs(predicted_dot_bracket))
    reference_pairs = _pair_set(dot_bracket_to_pairs(reference_dot_bracket))
    true_positive_pairs = sorted(predicted_pairs & reference_pairs)
    false_positive_pairs = sorted(predicted_pairs - reference_pairs)
    false_negative_pairs = sorted(reference_pairs - predicted_pairs)

    true_positives = len(true_positive_pairs)
    false_positives = len(false_positive_pairs)
    false_negatives = len(false_negative_pairs)

    return {
        "true_positive_pairs": true_positive_pairs,
        "false_positive_pairs": false_positive_pairs,
        "false_negative_pairs": false_negative_pairs,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {(int(pair[0]), int(pair[1])) for pair in pairs.tolist()}


def _add_comparison_legend(ax: Axes) -> None:
    handles = [
        Line2D([0], [0], color=PAIR_COMPARISON_COLORS["true_positive"], linewidth=1.2, label="TP"),
        Line2D([0], [0], color=PAIR_COMPARISON_COLORS["false_positive"], linewidth=1.2, label="FP"),
        Line2D(
            [0],
            [0],
            color=PAIR_COMPARISON_COLORS["false_negative"],
            linewidth=1.2,
            linestyle="--",
            label="FN",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7, handlelength=1.6)
