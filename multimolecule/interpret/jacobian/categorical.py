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
from typing import Any, Literal

import torch
from danling.data.utils import to_device
from torch import Tensor, nn

from ..outputs import JacobianOutput
from ..targets import ScalarTarget
from ..utils import (
    filter_forward_kwargs,
    get_input_embedding_layer,
    get_model_device,
    normalize_scalar_target,
    select_scalar_output,
)

JacobianReduction = Literal["none", "abs"]


def categorical_jacobian(
    model: nn.Module,
    input_ids: Tensor,
    target: ScalarTarget | int | None = None,
    *,
    positions: int | Sequence[int] | Tensor | None = None,
    top_k: int | None = None,
    reduction: JacobianReduction = "none",
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    **forward_kwargs: Any,
) -> JacobianOutput:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")

    embedding_layer = get_input_embedding_layer(model, usage="categorical Jacobians")
    device = get_model_device(model)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if position_ids is not None:
        position_ids = position_ids.to(device=device)
    forward_kwargs = to_device(forward_kwargs, device)
    target = normalize_scalar_target(target)
    captured_embeddings: Tensor | None = None

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
        nonlocal captured_embeddings
        if not isinstance(output, Tensor):
            raise TypeError(f"Embedding layer returned unsupported output type: {type(output)!r}")
        captured_embeddings = output
        captured_embeddings.retain_grad()

    handle = embedding_layer.register_forward_hook(hook)
    was_training = model.training
    kwargs = filter_forward_kwargs(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "return_dict": True,
            **forward_kwargs,
        },
    )
    model.eval()
    try:
        outputs = model(**kwargs)
        if captured_embeddings is None:
            raise RuntimeError("Failed to capture embedding activations from the model input embedding layer")
        scalar = select_scalar_output(outputs.logits, target, analysis_name="jacobian")
        score = scalar.sum()
        gradients = torch.autograd.grad(score, captured_embeddings, retain_graph=False, create_graph=False)[0]
    finally:
        handle.remove()
        model.train(was_training)

    embedding_weight = embedding_layer.weight.to(device=gradients.device, dtype=gradients.dtype)
    scores = torch.einsum("blh,vh->blv", gradients, embedding_weight)
    selected_positions = _normalize_positions(positions, input_ids.shape[1], input_ids.device)
    if selected_positions is not None:
        scores = scores.index_select(dim=1, index=selected_positions)

    if reduction == "abs":
        scores = scores.abs()
    elif reduction != "none":
        raise ValueError(f"Unsupported Jacobian reduction: {reduction}")

    top_k_indices = None
    top_k_scores = None
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        top_k = min(top_k, scores.shape[-1])
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=-1)

    return JacobianOutput(
        scores=scores,
        target=target,
        positions=selected_positions,
        reduction=reduction,
        top_k_indices=top_k_indices,
        top_k_scores=top_k_scores,
    )


def _normalize_positions(
    positions: int | Sequence[int] | Tensor | None, length: int, device: torch.device
) -> Tensor | None:
    if positions is None:
        return None
    if isinstance(positions, int):
        normalized = torch.tensor([positions], device=device, dtype=torch.long)
    elif isinstance(positions, Tensor):
        normalized = positions.to(device=device, dtype=torch.long).flatten()
    else:
        normalized = torch.tensor(list(positions), device=device, dtype=torch.long)
    if normalized.numel() == 0:
        raise ValueError("positions must contain at least one index")
    if normalized.min().item() < 0 or normalized.max().item() >= length:
        raise ValueError(f"positions must be within [0, {length})")
    return normalized


__all__ = ["JacobianReduction", "categorical_jacobian"]
