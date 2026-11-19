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
import torch
from torch import nn

from multimolecule.interpret.intervention import compare


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
        self.unused = nn.Identity()

    def forward(self, *, values, offset=0):
        return {"logits": self.features(values).sum(dim=-1) + offset}


def test_intervention_compares_model_scores_with_named_inputs():
    model = Prediction()
    inputs = {"values": torch.tensor([[1.0, 2.0], [2.0, -1.0]])}

    result = compare(
        model,
        inputs,
        replacements={model.features: torch.zeros_like},
        model_kwargs={"offset": 2},
    )

    torch.testing.assert_close(result.original, torch.tensor([5.0, 3.0]))
    torch.testing.assert_close(result.intervened, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(result.delta, torch.tensor([-3.0, -1.0]))
    torch.testing.assert_close(model(**inputs, offset=2)["logits"], result.original)


def test_intervention_removes_hooks_when_transform_raises():
    model = Prediction()
    inputs = {"values": torch.tensor([[1.0, 2.0]])}

    def fail(_):
        raise RuntimeError("failed transform")

    with pytest.raises(RuntimeError, match="failed transform"):
        compare(model, inputs, replacements={"features": fail})

    torch.testing.assert_close(model(**inputs)["logits"], torch.tensor([3.0]))


def test_intervention_requires_a_replacement():
    model = Prediction()
    inputs = {"values": torch.tensor([[1.0, 2.0]])}

    with pytest.raises(TypeError, match="replacement"):
        compare(model, inputs, replacements={"features": lambda _: None})


def test_intervention_reports_layers_skipped_by_forward():
    model = Prediction()
    inputs = {"values": torch.tensor([[1.0, 2.0]])}

    with pytest.raises(RuntimeError, match="unused"):
        compare(model, inputs, replacements={"unused": torch.zeros_like})


def test_intervention_rejects_two_replacements_for_the_same_module():
    model = Prediction()
    inputs = {"values": torch.tensor([[1.0, 2.0]])}

    with pytest.raises(ValueError, match="same submodule"):
        compare(model, inputs, replacements={"features": torch.zeros_like, model.features: torch.zeros_like})
