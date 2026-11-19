# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

"""Sparse autoencoders for captured model activations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["SparseAutoencoder", "SparseAutoencoderMetrics", "SparseAutoencoderOutput"]


@dataclass(frozen=True)
class SparseAutoencoderOutput:
    """A reconstruction and its corresponding nonnegative feature activations."""

    reconstruction: Tensor
    features: Tensor


@dataclass(frozen=True)
class SparseAutoencoderMetrics:
    """Objective, mean squared error, mean L1 penalty and mean active feature count."""

    loss: Tensor
    reconstruction_error: Tensor
    sparsity_penalty: Tensor
    mean_active_features: Tensor


class SparseAutoencoder(nn.Module):
    """Learn a sparse ReLU representation of tensors with features on the last axis.

    The encoder maps ``input_size`` channels to ``num_features`` sparse features, preserving
    all leading dimensions. The decoder normalizes weight columns during reconstruction
    to keep the sparsity penalty tied to feature activation magnitudes.

    Use ``loss`` with a PyTorch optimizer and save or restore weights with ``state_dict``
    and ``load_state_dict``.

    >>> sae = SparseAutoencoder(input_size=3, num_features=6)
    >>> optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    >>> optimizer.zero_grad()
    >>> sae.loss(torch.randn(4, 3)).backward()
    >>> optimizer.step()
    """

    def __init__(self, input_size: int, num_features: int) -> None:
        super().__init__()
        if input_size <= 0 or num_features <= 0:
            raise ValueError("input_size and num_features must be positive.")
        self.encoder = nn.Linear(input_size, num_features)
        self.decoder = nn.Linear(num_features, input_size)

    def encode(self, activations: Tensor) -> Tensor:
        """Encode activations as nonnegative features along the last axis."""
        input_size = self.encoder.in_features
        if activations.ndim == 0 or activations.shape[-1] != input_size:
            raise ValueError(
                f"Expected activations with a last dimension of {input_size}, got shape {tuple(activations.shape)}."
            )
        return self.encoder(activations - self.decoder.bias).relu()

    def decode(self, features: Tensor) -> Tensor:
        """Reconstruct activations from sparse features."""
        weight = F.normalize(self.decoder.weight, dim=0, eps=1e-6)
        return F.linear(features, weight, self.decoder.bias)

    def forward(self, activations: Tensor) -> SparseAutoencoderOutput:
        features = self.encode(activations)
        return SparseAutoencoderOutput(reconstruction=self.decode(features), features=features)

    def loss(self, activations: Tensor, *, sparsity_weight: float = 1e-3) -> Tensor:
        """Compute mean squared error plus weighted mean per-activation L1 penalty."""
        reconstruction_error, sparsity_penalty, _ = self._loss_terms(activations)
        return reconstruction_error + sparsity_weight * sparsity_penalty

    @torch.no_grad()
    def evaluate(self, activations: Tensor, *, sparsity_weight: float = 1e-3) -> SparseAutoencoderMetrics:
        """Measure reconstruction and sparsity across all activation rows."""
        reconstruction_error, sparsity_penalty, features = self._loss_terms(activations)
        return SparseAutoencoderMetrics(
            loss=reconstruction_error + sparsity_weight * sparsity_penalty,
            reconstruction_error=reconstruction_error,
            sparsity_penalty=sparsity_penalty,
            mean_active_features=features.count_nonzero(dim=-1).float().mean(),
        )

    def _loss_terms(self, activations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if activations.numel() == 0:
            raise ValueError("The SAE objective requires at least one activation.")
        output = self(activations)
        reconstruction_error = F.mse_loss(output.reconstruction, activations)
        sparsity_penalty = output.features.sum(dim=-1).mean()
        return reconstruction_error, sparsity_penalty, output.features
