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


from .backbones import BACKBONES, SEQUENCES, SequenceBackbone
from .criterions import CRITERIONS, Criterion
from .embeddings import POSITION_EMBEDDINGS, POSITION_EMBEDDINGS_HF, RotaryEmbedding, SinusoidalEmbedding
from .heads import (
    HEAD_TRANSFORMS,
    HEAD_TRANSFORMS_HF,
    HEADS,
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
from .necks import NECKS, CatNeck
from .networks import NETWORKS, ResNet, UNet
from .normlizations import LayerNorm2d
from .registry import MODELS

__all__ = [
    "MODELS",
    "MultiMoleculeModel",
    "POSITION_EMBEDDINGS",
    "POSITION_EMBEDDINGS_HF",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "BACKBONES",
    "SEQUENCES",
    "SequenceBackbone",
    "NECKS",
    "CatNeck",
    "BaseHeadConfig",
    "HeadConfig",
    "MaskedLMHeadConfig",
    "HEADS",
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
    "HEAD_TRANSFORMS",
    "HEAD_TRANSFORMS_HF",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
    "NETWORKS",
    "ResNet",
    "UNet",
    "CRITERIONS",
    "Criterion",
    "LayerNorm2d",
]
