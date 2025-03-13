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


from .backbones import BACKBONES
from .criterions import CRITERIONS, Criterion
from .embeddings import POSITION_EMBEDDINGS, POSITION_EMBEDDINGS_HF, RotaryEmbedding, SinusoidalEmbedding
from .heads import (
    HEAD_TRANSFORMS,
    HEAD_TRANSFORMS_HF,
    HEADS,
    BaseHeadConfig,
    BasePredictionHead,
    ContactAttentionLinearHead,
    ContactAttentionResnetHead,
    ContactLogitsResnetHead,
    ContactPredictionHead,
    HeadConfig,
    HeadOutput,
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
from .model import Model
from .necks import NECKS
from .normlizations import LayerNorm2D
from .registry import MODELS

__all__ = [
    "MODELS",
    "POSITION_EMBEDDINGS",
    "POSITION_EMBEDDINGS_HF",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "HEAD_TRANSFORMS",
    "HEAD_TRANSFORMS_HF",
    "CRITERIONS",
    "Model",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "BaseHeadConfig",
    "HeadConfig",
    "MaskedLMHeadConfig",
    "BasePredictionHead",
    "PredictionHead",
    "SequencePredictionHead",
    "TokenPredictionHead",
    "TokenKMerHead",
    "ContactPredictionHead",
    "ContactLogitsResnetHead",
    "ContactAttentionLinearHead",
    "ContactAttentionResnetHead",
    "MaskedLMHead",
    "HeadOutput",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
    "LayerNorm2D",
    "Criterion",
]
