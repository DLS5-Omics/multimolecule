# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from .criterions import Criterion
from .embeddings import PositionEmbeddingRegistry, PositionEmbeddingRegistryHF, RotaryEmbedding, SinusoidalEmbedding
from .heads import (
    ContactPredictionHead,
    HeadRegistry,
    HeadTransformRegistry,
    HeadTransformRegistryHF,
    IdentityTransform,
    LinearTransform,
    MaskedLMHead,
    NonLinearTransform,
    NucleotideHeadRegistryHF,
    NucleotideKMerHead,
    NucleotidePredictionHead,
    PredictionHead,
    SequencePredictionHead,
    TokenHeadRegistryHF,
    TokenKMerHead,
    TokenPredictionHead,
)

__all__ = [
    "Criterion",
    "PositionEmbeddingRegistry",
    "PositionEmbeddingRegistryHF",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "PredictionHead",
    "HeadRegistry",
    "SequencePredictionHead",
    "TokenHeadRegistryHF",
    "TokenPredictionHead",
    "TokenKMerHead",
    "NucleotideHeadRegistryHF",
    "NucleotidePredictionHead",
    "NucleotideKMerHead",
    "ContactPredictionHead",
    "MaskedLMHead",
    "HeadTransformRegistry",
    "HeadTransformRegistryHF",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
]
