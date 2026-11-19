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

from multimolecule.interpret.intervention import InterventionResult
from multimolecule.visualize.interpret import plot_intervention_comparison, plot_intervention_deltas


class TestPlotInterventionDeltas:
    def test_plots_named_signed_deltas(self, ax):
        deltas = {
            "Ablate A": -1.25,
            "Patch B": torch.tensor([0.5], dtype=torch.bfloat16),
            "Patch C": InterventionResult(torch.tensor(2.0), torch.tensor(4.0)),
        }

        assert plot_intervention_deltas(deltas, ax=ax) is ax

        assert [bar.get_width() for bar in ax.patches] == [-1.25, 0.5, 2.0]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == list(deltas)
        ax.figure.canvas.draw()

    def test_centers_zero(self, ax):
        plot_intervention_deltas({"Ablate": -1.25, "Patch": 2.0}, ax=ax)

        assert any(list(line.get_xdata()) == [0, 0] for line in ax.get_lines())
        lower, upper = ax.get_xlim()
        assert lower == -upper and upper >= 2

    def test_uses_custom_title(self, ax):
        plot_intervention_deltas({"Ablate": -1.0}, ax=ax, title="Conditions")

        assert ax.get_title() == "Conditions"

    def test_requires_one_score(self, ax):
        result = InterventionResult(torch.zeros(2), torch.ones(2))

        with pytest.raises(ValueError, match="one selected score"):
            plot_intervention_deltas({"Ablate": result}, ax=ax)


class TestPlotInterventionComparison:
    def test_pairs_original_and_intervened(self, ax):
        original = torch.tensor([1.0, -2.0], requires_grad=True)
        result = InterventionResult(original, torch.tensor([0.5, 3.0]))

        assert plot_intervention_comparison(result, sample_labels=["Sample A", "Sample B"], ax=ax) is ax

        heights = {group.get_label(): [bar.get_height() for bar in group] for group in ax.containers}
        assert heights == {"Original": [1.0, -2.0], "Intervened": [0.5, 3.0]}
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Sample A", "Sample B"]
        assert [label.get_text() for label in ax.get_legend().get_texts()] == ["Original", "Intervened"]
        ax.figure.canvas.draw()

    def test_plots_scalar_sample(self, ax):
        result = InterventionResult(torch.tensor(1.0), torch.tensor(-2.0))

        plot_intervention_comparison(result, ax=ax)

        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0"]

    def test_validates_sample_label_count(self, ax):
        result = InterventionResult(torch.tensor(1.0), torch.tensor(-2.0))

        with pytest.raises(ValueError, match="1 sample labels"):
            plot_intervention_comparison(result, sample_labels=["a", "b"], ax=ax)

    @pytest.mark.parametrize(
        ("original", "intervened"),
        [
            pytest.param(torch.ones(2, 1), torch.zeros(2, 1), id="unselected"),
            pytest.param(torch.ones(2), torch.zeros(3), id="mismatched"),
        ],
    )
    def test_rejects_invalid_score_shapes(self, ax, original, intervened):
        with pytest.raises(ValueError, match="matching scalar or 1D"):
            plot_intervention_comparison(InterventionResult(original, intervened), ax=ax)
