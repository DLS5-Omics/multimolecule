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


from .activation import capture_activations
from .attribution import AttributionOutput, attribute
from .jacobian import categorical_jacobian
from .outputs import ActivationOutput, AttentionOutput, JacobianOutput, SaeOutput
from .sae import run_sae
from .targets import ScalarTarget
from .visualization import format_topk_substitutions, plot_attention_map, plot_sae_features, plot_token_scores

__all__ = [
    "ActivationOutput",
    "AttentionOutput",
    "AttributionOutput",
    "JacobianOutput",
    "SaeOutput",
    "ScalarTarget",
    "attribute",
    "categorical_jacobian",
    "capture_activations",
    "format_topk_substitutions",
    "plot_attention_map",
    "plot_sae_features",
    "plot_token_scores",
    "run_sae",
]
