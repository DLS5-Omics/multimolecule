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

import numpy as np
import pytest
import torch

from multimolecule.interpret.sae import SparseAutoencoderOutput
from multimolecule.visualize.interpret import plot_attention_map, plot_jacobian, plot_sae_features, plot_token_scores


class TestPlotTokenScores:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
    def test_selects_batch(self, ax, dtype):
        scores = torch.tensor([[0.5, 1.5, 2.5], [1.0, -2.0, 3.0]], dtype=dtype)

        assert plot_token_scores(scores, batch_index=1, ax=ax) is ax

        np.testing.assert_array_equal(ax.images[0].get_array(), [[1.0, -2.0, 3.0]])

    def test_labels_tokens(self, ax):
        plot_token_scores(torch.ones(3), tokens=["a", "b", "c"], ax=ax)

        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b", "c"]


class TestPlotAttentionMap:
    def test_preserves_query_key_order(self, ax):
        weights = torch.arange(6).reshape(2, 3)

        assert plot_attention_map(weights, query_tokens=["q0", "q1"], key_tokens=["k0", "k1", "k2"], ax=ax) is ax

        np.testing.assert_array_equal(ax.images[0].get_array(), [[0, 1, 2], [3, 4, 5]])
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["q0", "q1"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["k0", "k1", "k2"]
        ax.figure.canvas.draw()

    def test_rejects_unselected_matrix(self, ax):
        with pytest.raises(ValueError, match="2D"):
            plot_attention_map(torch.ones(2, 2, 3, 3), ax=ax)


class TestPlotJacobian:
    def test_preserves_output_input_order(self, ax):
        matrix = torch.tensor([[-2.0, 0.0, 1.0], [3.0, 1.0, -4.0]], dtype=torch.bfloat16)

        assert plot_jacobian(matrix, input_labels=["x", "y", "z"], output_labels=[8, 9], ax=ax) is ax

        np.testing.assert_array_equal(ax.images[0].get_array(), [[-2.0, 0.0, 1.0], [3.0, 1.0, -4.0]])
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["x", "y", "z"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["8", "9"]
        ax.figure.canvas.draw()

    @pytest.mark.parametrize(
        ("values", "limit"),
        [
            pytest.param([[-2.0, 0.0, 1.0], [3.0, 1.0, -4.0]], 4.0, id="finite"),
            pytest.param([[float("nan"), float("inf"), -2.0, 1.0]], 2.0, id="nonfinite"),
        ],
    )
    def test_centers_color_scale(self, ax, values, limit):
        plot_jacobian(torch.tensor(values), ax=ax)

        image = ax.images[0]
        assert image.get_clim() == (-limit, limit)
        assert image.norm(0.0) == 0.5

    def test_centers_zero_matrix(self, ax):
        plot_jacobian(torch.zeros(1, 2), ax=ax)

        image = ax.images[0]
        lower, upper = image.get_clim()
        assert upper > 0
        assert lower == -upper
        assert image.norm(0.0) == 0.5

    def test_masks_nonfinite_values(self, ax):
        plot_jacobian(torch.tensor([[float("nan"), float("inf"), -2.0, 1.0]]), ax=ax)

        np.testing.assert_array_equal(np.ma.getmaskarray(ax.images[0].get_array()), [[True, True, False, False]])

    def test_rejects_unselected_matrix(self, ax):
        with pytest.raises(ValueError, match="2D"):
            plot_jacobian(torch.ones(2, 3, 4), ax=ax)


class TestPlotSaeFeatures:
    @pytest.mark.parametrize(
        ("feature_ids", "expected"),
        [
            pytest.param(None, ["2", "1"], id="default-ids"),
            pytest.param([10, 20, 30], ["30", "20"], id="custom-ids"),
        ],
    )
    def test_labels_top_features(self, ax, feature_ids, expected):
        features = torch.tensor([[0.1, 2.0, 3.0], [0.3, 1.0, 4.0]])
        output = SparseAutoencoderOutput(reconstruction=torch.zeros(2, 1), features=features)

        assert plot_sae_features(output, top_k=2, feature_ids=feature_ids, ax=ax) is ax

        np.testing.assert_array_equal(ax.images[0].get_array(), [[3.0, 4.0], [2.0, 1.0]])
        assert [tick.get_text() for tick in ax.get_yticklabels()] == expected
