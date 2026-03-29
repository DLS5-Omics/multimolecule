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

from transformers import AutoConfig, AutoModel

from .configuration_diffdock import DiffDockConfig
from .diffusion_utils import (
    GaussianFourierProjection,
    get_t_schedule,
    get_timestep_embedding,
    modify_conformer,
    set_time,
    sinusoidal_embedding,
    t_to_sigma,
)
from .geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from .graph_utils import knn_graph, radius, radius_graph, scatter, scatter_mean
from .layers import AtomEncoder, FCBlock, GaussianSmearing
from .modeling_diffdock import DiffDockConfidenceOutput, DiffDockModel, DiffDockPreTrainedModel, DiffDockScoreOutput
from .process_mols import build_complex_graph, build_ligand_graph, build_receptor_graph, lig_atom_featurizer
from .so3 import sample_vec as so3_sample_vec
from .so3 import score_norm as so3_score_norm
from .tensor_layers import TensorProductConvLayer, get_irrep_seq
from .torsion import get_transformation_mask, modify_conformer_torsion_angles

__all__ = [
    # Config
    "DiffDockConfig",
    # Model
    "DiffDockPreTrainedModel",
    "DiffDockModel",
    "DiffDockScoreOutput",
    "DiffDockConfidenceOutput",
    "TensorProductConvLayer",
    "get_irrep_seq",
    # Layers
    "AtomEncoder",
    "FCBlock",
    "GaussianSmearing",
    # Graph utilities (replaces torch_cluster + torch_scatter)
    "radius_graph",
    "radius",
    "knn_graph",
    "scatter",
    "scatter_mean",
    # Geometry
    "axis_angle_to_matrix",
    "rigid_transform_Kabsch_3D_torch",
    # Diffusion
    "t_to_sigma",
    "get_timestep_embedding",
    "get_t_schedule",
    "set_time",
    "modify_conformer",
    "sinusoidal_embedding",
    "GaussianFourierProjection",
    # SO(3)
    "so3_sample_vec",
    "so3_score_norm",
    # Torsion
    "get_transformation_mask",
    "modify_conformer_torsion_angles",
    # Data pipeline
    "build_ligand_graph",
    "build_receptor_graph",
    "build_complex_graph",
    "lig_atom_featurizer",
]

AutoConfig.register("diffdock", DiffDockConfig)
AutoModel.register(DiffDockConfig, DiffDockModel)
