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

"""Plot tensors from MultiMolecule's interpretability tools."""

from .heatmap import plot_attention_map, plot_jacobian, plot_sae_features, plot_token_scores
from .intervention import plot_intervention_comparison, plot_intervention_deltas

__all__ = [
    "plot_attention_map",
    "plot_intervention_comparison",
    "plot_intervention_deltas",
    "plot_jacobian",
    "plot_sae_features",
    "plot_token_scores",
]
