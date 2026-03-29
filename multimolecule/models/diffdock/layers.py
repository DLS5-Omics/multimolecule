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

"""Embedding and MLP layers for DiffDock.

Rewritten from the original DiffDock implementation without PyG dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


def FCBlock(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    layers: int,
    dropout: float,
    activation: str = "relu",
) -> nn.Sequential:
    """Fully connected MLP block.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.
        layers: Number of layers (must be >= 2).
        dropout: Dropout probability.
        activation: Activation function name.

    Returns:
        Sequential MLP.
    """
    act = ACTIVATIONS[activation]
    assert layers >= 2
    sequential: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act(), nn.Dropout(dropout)]
    for _ in range(layers - 2):
        sequential += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Dropout(dropout)]
    sequential += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*sequential)


class GaussianSmearing(nn.Module):
    """Gaussian radial basis function expansion for edge distances.

    Args:
        start: Minimum distance.
        stop: Maximum distance.
        num_gaussians: Number of Gaussian basis functions.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(nn.Module):
    """Encodes atom features (categorical + scalar) into a fixed-dim embedding.

    Args:
        emb_dim: Embedding output dimension.
        feature_dims: Tuple of (list of categorical feature sizes, num_scalar_features).
        sigma_embed_dim: Dimension of the timestep/sigma embedding appended to features.
        lm_embedding_dim: Dimension of language model embeddings (0 if not used).
    """

    def __init__(
        self,
        emb_dim: int,
        feature_dims: tuple[list[int], int],
        sigma_embed_dim: int,
        lm_embedding_dim: int = 0,
    ):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.additional_features_dim = feature_dims[1] + sigma_embed_dim + lm_embedding_dim
        for dim in feature_dims[0]:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.additional_features_dim > 0:
            self.additional_features_embedder = nn.Linear(self.additional_features_dim + emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_embedding: Tensor | int = 0
        assert x.shape[1] == self.num_categorical_features + self.additional_features_dim
        for i in range(self.num_categorical_features):
            x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i].long())

        if self.additional_features_dim > 0:
            x_embedding = self.additional_features_embedder(
                torch.cat([x_embedding, x[:, self.num_categorical_features :]], dim=1)
            )
        return x_embedding
