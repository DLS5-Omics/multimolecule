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

import subprocess
import sys
from textwrap import dedent

import pytest
import torch
from torch import nn


@pytest.fixture(scope="session")
def captum():
    return pytest.importorskip("captum")


@pytest.fixture
def named_score_batch():
    class Prediction(nn.Module):
        def forward(self, *, values, mask, auxiliary, position_scale, scale):
            score = (values * mask * position_scale).sum(dim=-1) + 3 * auxiliary[:, 0]
            return {"scores": scale * score}

    batch = {
        "values": torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
        "mask": torch.tensor([[True, False], [False, True]]),
        "auxiliary": torch.tensor([[5.0], [6.0]]),
        "position_scale": torch.tensor([[1.0, 4.0]]),
        "scale": torch.tensor(2.0),
    }
    return Prediction(), batch


@pytest.fixture
def without_captum():
    setup = dedent(
        """
        import sys

        attempts = []

        class MissingCaptum:
            @staticmethod
            def find_spec(fullname, path=None, target=None):
                if fullname == "captum":
                    attempts.append(fullname)
                    raise ModuleNotFoundError("No module named 'captum'", name=fullname)

        sys.meta_path.insert(0, MissingCaptum())
        """
    )

    def run(script):
        return subprocess.run(
            [sys.executable, "-c", setup + dedent(script)], capture_output=True, text=True, check=False
        )

    return run


@pytest.fixture
def weighted_relu():
    class Score(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.readout = nn.Identity()

        def forward(self, inputs):
            weights = inputs.new_tensor([-2.0, 3.0, 4.0])
            return self.readout((self.relu(inputs) * weights).sum(dim=-1, keepdim=True))

    return Score()


@pytest.fixture
def epsilon_rule(captum):
    # Captum documents custom LRP rules through layer.rule. It ships its reference rules here.
    from captum.attr._utils.lrp_rules import EpsilonRule

    return EpsilonRule


@pytest.fixture
def mutating_lrp_model(epsilon_rule):
    class DoubledWeights(epsilon_rule):
        def _manipulate_weights(self, module, inputs, outputs):
            with torch.no_grad():
                module.weight.mul_(2)

    linear = nn.Linear(2, 1, bias=False)
    nn.init.ones_(linear.weight)
    linear.rule = DoubledWeights()
    return nn.Sequential(linear)


@pytest.fixture
def lrp_context():
    class Score(nn.Module):
        def __init__(self):
            super().__init__()
            self.primary = nn.Linear(2, 1, bias=False)
            self.context = nn.Linear(2, 1, bias=False)
            nn.init.ones_(self.primary.weight)
            nn.init.ones_(self.context.weight)

        def forward(self, *, values, context):
            return self.primary(values) + self.context(context)

    batch = {"values": torch.ones(2, 2), "context": torch.tensor([[2.0, 3.0]])}
    return Score(), batch
