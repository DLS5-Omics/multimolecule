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

from __future__ import annotations

from dataclasses import dataclass

from torch import FloatTensor
from transformers.modeling_outputs import ModelOutput


@dataclass
class HeadOutput(ModelOutput):
    r"""
    Output of a prediction head.

    Args:
        logits: The prediction logits from the head.
        loss: The loss from the head.
            Defaults to None.
    """

    logits: FloatTensor
    loss: FloatTensor | None = None
