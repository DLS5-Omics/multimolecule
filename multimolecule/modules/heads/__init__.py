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


from .config import BaseHeadConfig, HeadConfig, MaskedLMHeadConfig
from .contact import (
    ContactAttentionHead,
    ContactAttentionResNetHead,
    ContactAttentionUNetHead,
    ContactPredictionHead,
    ContactPredictionResNetHead,
    ContactPredictionUNetHead,
)
from .generic import BasePredictionHead, PredictionHead
from .output import HeadOutput
from .pretrain import MaskedLMHead
from .registry import HEAD_TRANSFORMS, HEAD_TRANSFORMS_HF, HEADS
from .sequence import SequencePredictionHead
from .token import TokenKMerHead, TokenPredictionHead
from .transform import IdentityTransform, LinearTransform, NonLinearTransform

__all__ = [
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
]
