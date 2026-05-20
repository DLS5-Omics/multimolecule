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

import pytest

import multimolecule  # noqa: F401
from multimolecule import (
    A2zChromatinConfig,
    A2zChromatinForSequencePrediction,
    AutoModelForMethylationPrediction,
    BassetConfig,
    DeepCpgDnaConfig,
    DeepCpgDnaForSequencePrediction,
    MmSpliceConfig,
)


def test_methylation_auto_class_keeps_domain_specific_heads():
    assert isinstance(
        AutoModelForMethylationPrediction.from_config(DeepCpgDnaConfig()), DeepCpgDnaForSequencePrediction
    )
    assert isinstance(
        AutoModelForMethylationPrediction.from_config(A2zChromatinConfig()), A2zChromatinForSequencePrediction
    )


@pytest.mark.parametrize("config", [BassetConfig(), MmSpliceConfig()])
def test_methylation_auto_class_rejects_non_methylation_models(config):
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForMethylationPrediction.from_config(config)
