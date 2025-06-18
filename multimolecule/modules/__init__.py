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


from .criterions import Criterion, CriterionRegistry
from .embeddings import PositionEmbeddingRegistry, RotaryEmbedding, SinusoidalEmbedding
from .heads import (
    BaseHeadConfig,
    BasePredictionHead,
    ContactAttentionHead,
    ContactAttentionResNetHead,
    ContactAttentionUNetHead,
    ContactPredictionHead,
    ContactPredictionResNetHead,
    ContactPredictionUNetHead,
    HeadConfig,
    HeadOutput,
    HeadRegistry,
    HeadTransformRegistry,
    HeadTransformRegistryHF,
    IdentityTransform,
    LinearTransform,
    MaskedLMHead,
    MaskedLMHeadConfig,
    NonLinearTransform,
    PredictionHead,
    SequencePredictionHead,
    TokenKMerHead,
    TokenPredictionHead,
)
from .model import MultiMoleculeModel
from .normlizations import LayerNorm2d
from .registry import ModelRegistry

__all__ = [
    "ModelRegistry",
    "MultiMoleculeModel",
    "CriterionRegistry",
    "Criterion",
    "PositionEmbeddingRegistry",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "BaseHeadConfig",
    "HeadConfig",
    "MaskedLMHeadConfig",
    "HeadRegistry",
    "BasePredictionHead",
    "PredictionHead",
    "SequencePredictionHead",
    "TokenPredictionHead",
    "TokenKMerHead",
    "ContactPredictionHead",
    "ContactPredictionResNetHead",
    "ContactPredictionUNetHead",
    "ContactAttentionHead",
    "ContactAttentionResNetHead",
    "ContactAttentionUNetHead",
    "MaskedLMHead",
    "HeadOutput",
    "HeadTransformRegistry",
    "HeadTransformRegistryHF",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
    "LayerNorm2d",
]
