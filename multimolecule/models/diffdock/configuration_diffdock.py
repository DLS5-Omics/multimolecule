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


from __future__ import annotations

from ..configuration_utils import PreTrainedConfig


class DiffDockConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`DiffDockModel`][multimolecule.models.DiffDockModel]. It is used to instantiate a DiffDock model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the DiffDock
    [gcorso/DiffDock](https://github.com/gcorso/DiffDock) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        in_lig_edge_features:
            Number of ligand bond type features (one-hot encoded bond types).
        sigma_embed_dim:
            Base dimension of the timestep/noise-level embedding.
        sh_lmax:
            Maximum degree of spherical harmonics used in tensor products.
        ns:
            Number of scalar (invariant) features in equivariant representations.
        nv:
            Number of vector features in equivariant representations.
        num_conv_layers:
            Number of equivariant message-passing convolution layers.
        lig_max_radius:
            Cutoff radius (Angstroms) for ligand intra-molecular graph.
        rec_max_radius:
            Cutoff radius (Angstroms) for receptor residue contact graph.
        cross_max_distance:
            Cutoff distance (Angstroms) for ligand-receptor cross edges.
        center_max_distance:
            Maximum distance for center-to-ligand edges (TR/ROT prediction).
        distance_embed_dim:
            Number of Gaussian basis functions for distance embedding.
        cross_distance_embed_dim:
            Number of Gaussian basis functions for cross-edge distance embedding.
        no_torsion:
            If True, skip torsion angle prediction.
        scale_by_sigma:
            Scale predicted scores by the noise level sigma.
        norm_by_sigma:
            Normalize scores by sigma.
        use_second_order_repr:
            Use second-order (l=2) spherical harmonics in representations.
        batch_norm:
            Apply equivariant batch normalization in convolution layers.
        dynamic_max_cross:
            Use sigma-dependent cross-edge distance cutoff.
        dropout:
            Dropout probability in MLP layers.
        smooth_edges:
            Apply smooth cosine edge weighting by distance.
        odd_parity:
            Use odd-parity output irreps for score prediction.
        separate_noise_schedule:
            Use independent noise schedules for translation, rotation, and torsion.
        confidence_mode:
            If True, predict docking confidence instead of scores.
        confidence_dropout:
            Dropout for the confidence prediction head.
        confidence_no_batchnorm:
            Disable batch normalization in the confidence prediction head.
        num_prot_emb_layers:
            Number of protein-only embedding layers before joint convolution.
        reduce_pseudoscalars:
            Reduce pseudoscalar feature dimension to nv instead of ns.
        differentiate_convolutions:
            Use separate FC weight networks per edge group type.
        tp_weights_layers:
            Number of MLP layers in the tensor product weight network.
        num_confidence_outputs:
            Number of confidence output values per complex.
        atom_confidence:
            Predict per-atom confidence in addition to per-complex.
        atom_num_confidence_outputs:
            Number of per-atom confidence output values.
        fixed_center_conv:
            Use fixed center for the final TR/ROT convolution.
        embed_also_ligand:
            Apply additional embedding layers to the ligand before joint convolution.
        lm_embedding_type:
            Type of protein language model embedding ("precomputed" or None).

    Examples:
        >>> from multimolecule import DiffDockConfig, DiffDockModel
        >>> # Initializing a DiffDock gcorso/DiffDock style configuration
        >>> configuration = DiffDockConfig()
        >>> # Initializing a model (with random weights) from the gcorso/DiffDock style configuration
        >>> model = DiffDockModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "diffdock"

    def __init__(
        self,
        in_lig_edge_features: int = 4,
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        ns: int = 16,
        nv: int = 4,
        num_conv_layers: int = 2,
        lig_max_radius: float = 5.0,
        rec_max_radius: float = 30.0,
        cross_max_distance: float = 250.0,
        center_max_distance: float = 30.0,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        no_torsion: bool = False,
        scale_by_sigma: bool = True,
        norm_by_sigma: bool = True,
        use_second_order_repr: bool = False,
        batch_norm: bool = True,
        dynamic_max_cross: bool = False,
        dropout: float = 0.0,
        smooth_edges: bool = False,
        odd_parity: bool = False,
        separate_noise_schedule: bool = False,
        confidence_mode: bool = False,
        confidence_dropout: float = 0.0,
        confidence_no_batchnorm: bool = False,
        num_prot_emb_layers: int = 0,
        reduce_pseudoscalars: bool = False,
        differentiate_convolutions: bool = True,
        tp_weights_layers: int = 2,
        num_confidence_outputs: int = 1,
        atom_confidence: bool = False,
        atom_num_confidence_outputs: int = 1,
        fixed_center_conv: bool = False,
        embed_also_ligand: bool = False,
        lm_embedding_type: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sh_lmax = sh_lmax
        self.ns = ns
        self.nv = nv
        self.num_conv_layers = num_conv_layers
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.no_torsion = no_torsion
        self.scale_by_sigma = scale_by_sigma
        self.norm_by_sigma = norm_by_sigma
        self.use_second_order_repr = use_second_order_repr
        self.batch_norm = batch_norm
        self.dynamic_max_cross = dynamic_max_cross
        self.dropout = dropout
        self.smooth_edges = smooth_edges
        self.odd_parity = odd_parity
        self.separate_noise_schedule = separate_noise_schedule
        self.confidence_mode = confidence_mode
        self.confidence_dropout = confidence_dropout
        self.confidence_no_batchnorm = confidence_no_batchnorm
        self.num_prot_emb_layers = num_prot_emb_layers
        self.reduce_pseudoscalars = reduce_pseudoscalars
        self.differentiate_convolutions = differentiate_convolutions
        self.tp_weights_layers = tp_weights_layers
        self.num_confidence_outputs = num_confidence_outputs
        self.atom_confidence = atom_confidence
        self.atom_num_confidence_outputs = atom_num_confidence_outputs
        self.fixed_center_conv = fixed_center_conv
        self.embed_also_ligand = embed_also_ligand
        self.lm_embedding_type = lm_embedding_type
