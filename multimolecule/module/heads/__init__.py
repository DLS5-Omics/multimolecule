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
from .generic import ClassificationHead
from .nuleotide import NucleotideClassificationHead, NucleotideHeads, NucleotideKMerHead
from .pretrain import MaskedLMHead
from .sequence import SequenceClassificationHead
from .token import TokenClassificationHead, TokenHeads, TokenKMerHead
from .transform import HeadTransforms, IdentityTransform, LinearTransform, NonLinearTransform

__all__ = [
    "ClassificationHead",
    "SequenceClassificationHead",
    "TokenHeads",
    "TokenClassificationHead",
    "TokenKMerHead",
    "NucleotideHeads",
    "NucleotideClassificationHead",
    "NucleotideKMerHead",
    "ContactPredictionHead",
    "MaskedLMHead",
    "HeadTransforms",
    "LinearTransform",
    "NonLinearTransform",
    "IdentityTransform",
]
