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
    ContactAttentionLinearHead,
    ContactAttentionResnetHead,
    ContactLogitsResnetHead,
    ContactPredictionHead,
)
from .generic import BasePredictionHead, PredictionHead
from .output import HeadOutput
from .pretrain import MaskedLMHead
from .registry import HeadRegistry
from .sequence import SequencePredictionHead
from .token import TokenKMerHead, TokenPredictionHead
from .transform import (
    HeadTransformRegistry,
    HeadTransformRegistryHF,
    IdentityTransform,
    LinearTransform,
    NonLinearTransform,
)

__all__ = [
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
    "ContactLogitsResnetHead",
    "ContactAttentionLinearHead",
    "ContactAttentionResnetHead",
    "MaskedLMHead",
    "HeadOutput",
    "HeadTransformRegistry",
    "HeadTransformRegistryHF",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
]
