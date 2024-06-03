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

from .contact import ContactPredictionHead
from .generic import PredictionHead
from .nuleotide import NucleotideHeadRegistryHF, NucleotideKMerHead, NucleotidePredictionHead
from .pretrain import MaskedLMHead
from .registry import HeadRegistry
from .sequence import SequencePredictionHead
from .token import TokenHeadRegistryHF, TokenKMerHead, TokenPredictionHead
from .transform import (
    HeadTransformRegistry,
    HeadTransformRegistryHF,
    IdentityTransform,
    LinearTransform,
    NonLinearTransform,
)

__all__ = [
    "HeadRegistry",
    "PredictionHead",
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
