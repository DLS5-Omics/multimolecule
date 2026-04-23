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

from collections.abc import Sequence

import pandas as pd
from torch import Tensor

from ..outputs import JacobianOutput


def format_topk_substitutions(
    output: JacobianOutput,
    *,
    vocabulary: Sequence[str] | None = None,
    batch_index: int = 0,
    top_k: int | None = None,
) -> pd.DataFrame:
    if output.top_k_indices is not None and output.top_k_scores is not None:
        indices = output.top_k_indices[batch_index].detach().cpu()
        scores = output.top_k_scores[batch_index].detach().cpu()
    else:
        if top_k is None:
            raise ValueError("top_k must be provided when the Jacobian output does not already include top-k results")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        top_k = min(top_k, output.scores.shape[-1])
        values, indices = output.scores[batch_index].detach().cpu().topk(k=top_k, dim=-1)
        scores = values

    position_ids = (
        output.positions.detach().cpu().tolist()
        if isinstance(output.positions, Tensor)
        else list(range(indices.shape[0]))
    )
    rows: list[dict[str, object]] = []
    for row_index, position in enumerate(position_ids):
        for rank in range(indices.shape[1]):
            token_index = int(indices[row_index, rank].item())
            rows.append(
                {
                    "position": int(position),
                    "rank": rank + 1,
                    "token_index": token_index,
                    "token": _lookup_vocabulary(vocabulary, token_index),
                    "score": float(scores[row_index, rank].item()),
                }
            )
    return pd.DataFrame(rows)


def _lookup_vocabulary(vocabulary: Sequence[str] | None, token_index: int) -> str:
    if vocabulary is None:
        return str(token_index)
    if token_index >= len(vocabulary):
        raise ValueError(
            f"Vocabulary lookup failed for token index {token_index} with vocabulary size {len(vocabulary)}"
        )
    return str(vocabulary[token_index])


__all__ = ["format_topk_substitutions"]
