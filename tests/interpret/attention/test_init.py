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

from multimolecule.interpret import attention


class TestCapture:
    def test_selects_layers_and_heads(self):
        class Model(nn.Module):
            def forward(self, values, *, output_attentions=False):
                return {"attentions": (values, values + 100) if output_attentions else None}

        values = torch.arange(48.0).reshape(2, 2, 3, 2, 2)
        maps = attention.capture(
            Model(),
            values,
            layers=[-1, 0],
            heads=[2, 0],
            head_dim=2,
            model_kwargs={"output_attentions": True},
        )

        torch.testing.assert_close(maps, (values[:, :, [2, 0]] + 100, values[:, :, [2, 0]]))

    def test_keeps_single_head_axis(self):
        class Model(nn.Module):
            def forward(self, values):
                return {"weights": values}

        values = torch.arange(12.0).reshape(1, 3, 2, 2)
        maps = attention.capture(Model(), values, heads=-1, output=lambda result: result["weights"])

        torch.testing.assert_close(maps, (values[:, -1:],))

    def test_reports_missing_attention(self):
        with pytest.raises(ValueError, match="output_attentions"):
            attention.capture(nn.Identity(), torch.zeros(1, 2, 3))


class TestMeanHeads:
    def test_preserves_other_axes(self):
        first = torch.arange(24.0).reshape(1, 2, 3, 2, 2)
        second = torch.arange(8.0).reshape(1, 1, 2, 2, 2)

        averaged = attention.mean_heads((first, second), head_dim=2)

        torch.testing.assert_close(averaged, (first.mean(dim=2), second.mean(dim=2)))


class TestMeanLayers:
    def test_keeps_head_axis(self):
        first = torch.arange(8.0).reshape(1, 2, 2, 2)
        second = first + 2

        averaged = attention.mean_layers((first, second))

        torch.testing.assert_close(averaged, first + 1)

    def test_requires_matching_shapes(self):
        weights = (torch.ones(1, 2, 2, 2), torch.ones(1, 3, 2, 2))
        with pytest.raises(ValueError, match="same shape"):
            attention.mean_layers(weights)


class TestRollout:
    @pytest.mark.parametrize("head_dim", [0, 1, -1], ids=["first", "second", "last"])
    def test_composes_layers_in_order(self, head_dim):
        earlier = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]])
        later = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        weights = [layer.movedim(1, head_dim) for layer in (earlier, later)]

        result = attention.rollout(weights, head_dim=head_dim)

        expected = torch.tensor([[[0.75, 0.25], [0.375, 0.625]]])
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("shape", [(1, 2, 2, 3), (1, 2, 2, 2, 2)], ids=["rectangular", "extra-axis"])
    def test_requires_square_attention(self, shape):
        with pytest.raises(ValueError, match="square"):
            attention.rollout((torch.ones(shape),))


class TestAttentionAggregation:
    @pytest.mark.parametrize("aggregate", ["mean_heads", "mean_layers", "rollout"])
    def test_requires_layers(self, aggregate):
        with pytest.raises(ValueError, match="at least one"):
            getattr(attention, aggregate)(())
