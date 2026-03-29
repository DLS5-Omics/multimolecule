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

"""DiffDock all-atom model, rewritten without PyG.

Replaces:
- torch_geometric.data.HeteroData -> plain dict
- torch_cluster.radius / radius_graph -> graph_utils.radius / radius_graph
- torch_scatter.scatter_mean -> graph_utils.scatter_mean
- PyG DataLoader / Batch -> NestedTensor-based batching

The model operates on a dict-based data structure. Each sample is a dict with keys:
    - 'ligand_x': Tensor (N_lig, F_lig)    - ligand node features
    - 'ligand_pos': Tensor (N_lig, 3)       - ligand atom positions
    - 'ligand_edge_index': Tensor (2, E_lig) - ligand bond edges
    - 'ligand_edge_attr': Tensor (E_lig, F_edge) - ligand edge features
    - 'ligand_edge_mask': Tensor (E_lig,)    - mask for rotatable bonds
    - 'ligand_mask_rotate': Tensor            - rotation masks for torsion angles
    - 'receptor_x': Tensor (N_rec, F_rec)   - receptor residue features
    - 'receptor_pos': Tensor (N_rec, 3)      - receptor C-alpha positions
    - 'receptor_edge_index': Tensor (2, E_rec)
    - 'atom_x': Tensor (N_atom, F_atom)     - receptor all-atom features
    - 'atom_pos': Tensor (N_atom, 3)         - receptor all-atom positions
    - 'atom_edge_index': Tensor (2, E_atom)
    - 'atom_receptor_edge_index': Tensor (2, E_ar)
    - 'complex_t': dict with 'tr', 'rot', 'tor' timestep tensors
    - 'ligand_batch': Tensor (N_lig,)        - batch assignment
    - 'receptor_batch': Tensor (N_rec,)
    - 'atom_batch': Tensor (N_atom,)
    - 'num_graphs': int
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from e3nn import o3
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_diffdock import DiffDockConfig
from .graph_utils import radius, radius_graph, scatter_mean
from .layers import AtomEncoder, GaussianSmearing
from .tensor_layers import TensorProductConvLayer, get_irrep_seq

# Feature dimension constants (from original DiffDock process_mols.py)
lig_feature_dims = (
    [119, 4, 12, 16, 8, 9, 5, 6, 2, 7, 2, 2, 2, 2, 2, 2],
    0,
)
rec_residue_feature_dims = ([38], 0)
rec_atom_feature_dims = ([38, 119, 20, 53], 0)

AGGREGATORS: dict[str, Callable] = {
    "mean": lambda x: torch.mean(x, dim=1),
    "max": lambda x: torch.max(x, dim=1)[0],
    "min": lambda x: torch.min(x, dim=1)[0],
    "std": lambda x: torch.std(x, dim=1),
}


@dataclass
class DiffDockScoreOutput(ModelOutput):
    """
    Output type for DiffDock score prediction.

    Args:
        tr_score (`torch.FloatTensor` of shape `(batch_size, 3)`):
            Translational score vectors.
        rot_score (`torch.FloatTensor` of shape `(batch_size, 3)`):
            Rotational score vectors.
        tor_score (`torch.FloatTensor` of shape `(num_torsions,)`):
            Torsional score values for each rotatable bond.
    """

    tr_score: torch.FloatTensor | None = None
    rot_score: torch.FloatTensor | None = None
    tor_score: torch.FloatTensor | None = None


@dataclass
class DiffDockConfidenceOutput(ModelOutput):
    """
    Output type for DiffDock confidence prediction.

    Args:
        confidence (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, num_outputs)`):
            Per-complex confidence scores.
        atom_confidence (`torch.FloatTensor` of shape `(num_ligand_atoms,)`):
            Per-atom confidence scores.
    """

    confidence: torch.FloatTensor | None = None
    atom_confidence: torch.FloatTensor | None = None


class DiffDockPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DiffDockConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["TensorProductConvLayer"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class DiffDockModel(DiffDockPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import DiffDockConfig, DiffDockModel
        >>> config = DiffDockConfig()
        >>> model = DiffDockModel(config)
    """

    def __init__(
        self,
        config: DiffDockConfig,
        t_to_sigma: Callable | None = None,
        timestep_emb_func: Callable | None = None,
    ):
        super().__init__(config)

        # Runtime callables (not part of config, set externally)
        self.t_to_sigma = t_to_sigma
        self.timestep_emb_func = timestep_emb_func

        # Unpack config
        in_lig_edge_features = config.in_lig_edge_features
        sigma_embed_dim = config.sigma_embed_dim * (3 if config.separate_noise_schedule else 1)
        sh_lmax = config.sh_lmax
        ns = config.ns
        nv = config.nv
        num_conv_layers = config.num_conv_layers
        dropout = config.dropout
        batch_norm = config.batch_norm
        use_second_order_repr = config.use_second_order_repr
        num_prot_emb_layers = config.num_prot_emb_layers
        differentiate_convolutions = config.differentiate_convolutions
        tp_weights_layers = config.tp_weights_layers
        reduce_pseudoscalars = config.reduce_pseudoscalars
        confidence_mode = config.confidence_mode
        confidence_dropout = config.confidence_dropout
        confidence_no_batchnorm = config.confidence_no_batchnorm
        num_confidence_outputs = config.num_confidence_outputs
        atom_confidence = config.atom_confidence
        atom_num_confidence_outputs = config.atom_num_confidence_outputs
        odd_parity = config.odd_parity
        embed_also_ligand = config.embed_also_ligand
        lm_embedding_type = config.lm_embedding_type

        distance_embed_dim = config.distance_embed_dim
        cross_distance_embed_dim = config.cross_distance_embed_dim
        lig_max_radius = config.lig_max_radius
        rec_max_radius = config.rec_max_radius
        cross_max_distance = config.cross_max_distance
        center_max_distance = config.center_max_distance
        no_torsion = config.no_torsion

        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = config.dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = config.scale_by_sigma
        self.norm_by_sigma = config.norm_by_sigma
        self.no_torsion = config.no_torsion
        self.smooth_edges = config.smooth_edges
        self.odd_parity = odd_parity
        self.num_conv_layers = num_conv_layers
        self.separate_noise_schedule = config.separate_noise_schedule
        self.confidence_mode = confidence_mode
        self.num_prot_emb_layers = num_prot_emb_layers
        self.differentiate_convolutions = differentiate_convolutions
        self.reduce_pseudoscalars = reduce_pseudoscalars
        self.atom_confidence = atom_confidence
        self.atom_num_confidence_outputs = atom_num_confidence_outputs
        self.fixed_center_conv = config.fixed_center_conv

        lm_embedding_dim = 0
        if lm_embedding_type == "precomputed":
            lm_embedding_dim = 1280

        # Embedding layers
        self.lig_node_embedding = AtomEncoder(
            emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim
        )
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.rec_sigma_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.rec_node_embedding = AtomEncoder(
            emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=0, lm_embedding_dim=lm_embedding_dim,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.atom_node_embedding = AtomEncoder(
            emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=0,
        )
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
        )

        # Distance expansions
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)

        # Protein embedding layers
        rec_emb_layers = []
        for i in range(num_prot_emb_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                n_edge_features=3 * ns, hidden_features=3 * ns, residual=True,
                batch_norm=batch_norm, dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else 4,
            )
            rec_emb_layers.append(layer)
        self.rec_emb_layers = nn.ModuleList(rec_emb_layers)

        self.embed_also_ligand = embed_also_ligand
        if embed_also_ligand:
            lig_emb_layers = []
            for i in range(num_prot_emb_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
                layer = TensorProductConvLayer(
                    in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                    n_edge_features=3 * ns, hidden_features=3 * ns, residual=True,
                    batch_norm=batch_norm, dropout=dropout,
                    faster=sh_lmax == 1 and not use_second_order_repr,
                    tp_weights_layers=tp_weights_layers, edge_groups=1,
                )
                lig_emb_layers.append(layer)
            self.lig_emb_layers = nn.ModuleList(lig_emb_layers)

        # Main convolution layers
        conv_layers = []
        for i in range(num_prot_emb_layers, num_prot_emb_layers + num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                n_edge_features=3 * ns, hidden_features=3 * ns, residual=True,
                batch_norm=batch_norm, dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else (
                    3 if i == num_prot_emb_layers + num_conv_layers - 1 else 9
                ),
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        # Output heads
        if confidence_mode:
            output_confidence_dim = num_confidence_outputs
            input_size = ns + (nv if reduce_pseudoscalars else ns) \
                if num_conv_layers + num_prot_emb_layers >= 3 else ns

            if atom_confidence:
                self.atom_confidence_predictor = nn.Sequential(
                    nn.Linear(input_size, ns),
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(), nn.Dropout(confidence_dropout),
                    nn.Linear(ns, ns),
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(), nn.Dropout(confidence_dropout),
                    nn.Linear(ns, atom_num_confidence_outputs + ns),
                )
                input_size = ns

            self.confidence_predictor = nn.Sequential(
                nn.Linear(input_size, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(), nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(), nn.Dropout(confidence_dropout),
                nn.Linear(ns, output_confidence_dim),
            )
        else:
            # Score prediction heads
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
            )
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps="2x1o + 2x1e" if not odd_parity else "1x1o + 1x1e",
                n_edge_features=2 * ns, residual=False, dropout=dropout, batch_norm=batch_norm,
            )
            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1),
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1),
            )

            if not no_torsion:
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns),
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e" if not odd_parity else f"{ns}x0o",
                    n_edge_features=3 * ns, residual=False, dropout=dropout, batch_norm=batch_norm,
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not odd_parity else ns, ns, bias=False),
                    nn.Tanh(), nn.Dropout(dropout), nn.Linear(ns, 1, bias=False),
                )

        # Initialize weights and apply final processing
        self.post_init()

    def get_edge_weight(self, edge_vec: Tensor, max_norm: float | Tensor) -> Tensor | float:
        if self.smooth_edges:
            normalised_norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    def build_lig_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build ligand intra-molecular graph with radius edges."""
        if self.separate_noise_schedule:
            node_sigma_emb = torch.cat(
                [self.timestep_emb_func(data["ligand_node_t"][k]) for k in ("tr", "rot", "tor")], dim=1
            )
        else:
            node_sigma_emb = self.timestep_emb_func(data["ligand_node_t"]["tr"])
        data["ligand_node_sigma_emb"] = node_sigma_emb

        radius_edges = radius_graph(data["ligand_pos"], self.lig_max_radius, data["ligand_batch"])
        edge_index = torch.cat([data["ligand_edge_index"], radius_edges], dim=1).long()
        edge_attr = torch.cat([
            data["ligand_edge_attr"],
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data["ligand_pos"].device),
        ], dim=0)

        edge_sigma_emb = node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], dim=1)
        node_attr = torch.cat([data["ligand_x"], node_sigma_emb], dim=1)

        src, dst = edge_index
        edge_vec = data["ligand_pos"][dst.long()] - data["ligand_pos"][src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], dim=1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build receptor residue graph from precomputed edges."""
        node_attr = data["receptor_x"]
        edge_index = data["receptor_edge_index"]
        src, dst = edge_index
        edge_vec = data["receptor_pos"][dst.long()] - data["receptor_pos"][src.long()]
        edge_attr = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)
        return node_attr, edge_attr, edge_sh, edge_weight

    def build_atom_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build receptor all-atom graph from precomputed edges."""
        node_attr = data["atom_x"]
        edge_index = data["atom_edge_index"]
        src, dst = edge_index
        edge_vec = data["atom_pos"][dst.long()] - data["atom_pos"][src.long()]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return node_attr, edge_attr, edge_sh, edge_weight

    def build_cross_rec_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build atom-receptor cross edges from precomputed edges."""
        ar_edge_index = data["atom_receptor_edge_index"]
        ar_edge_vec = data["receptor_pos"][ar_edge_index[1].long()] - data["atom_pos"][ar_edge_index[0].long()]
        ar_edge_attr = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization="component")
        return ar_edge_attr, ar_edge_sh, 1

    def build_cross_lig_conv_graph(
        self, data: dict[str, Any], lr_cross_distance_cutoff: float | Tensor
    ) -> tuple[Tensor, ...]:
        """Build ligand-receptor and ligand-atom cross edges dynamically."""
        # Ligand to Receptor
        if torch.is_tensor(lr_cross_distance_cutoff):
            lr_edge_index = radius(
                data["receptor_pos"] / lr_cross_distance_cutoff[data["receptor_batch"]],
                data["ligand_pos"] / lr_cross_distance_cutoff[data["ligand_batch"]],
                1, data["receptor_batch"], data["ligand_batch"], max_num_neighbors=10000,
            )
        else:
            lr_edge_index = radius(
                data["receptor_pos"], data["ligand_pos"], lr_cross_distance_cutoff,
                data["receptor_batch"], data["ligand_batch"], max_num_neighbors=10000,
            )

        lr_edge_vec = data["receptor_pos"][lr_edge_index[1].long()] - data["ligand_pos"][lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data["ligand_node_sigma_emb"][lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], dim=1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization="component")

        cutoff_d = lr_cross_distance_cutoff[data["ligand_batch"][lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        # Ligand to Atom
        la_edge_index = radius(
            data["atom_pos"], data["ligand_pos"], self.lig_max_radius,
            data["atom_batch"], data["ligand_batch"], max_num_neighbors=10000,
        )
        la_edge_vec = data["atom_pos"][la_edge_index[1].long()] - data["ligand_pos"][la_edge_index[0].long()]
        la_edge_length_emb = self.lig_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data["ligand_node_sigma_emb"][la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], dim=1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization="component")
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        return (lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight,
                la_edge_index, la_edge_attr, la_edge_sh, la_edge_weight)

    def build_center_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build center-to-ligand graph for TR/ROT prediction."""
        edge_index = torch.cat([
            data["ligand_batch"].unsqueeze(0),
            torch.arange(len(data["ligand_batch"]), device=data["ligand_pos"].device).unsqueeze(0),
        ], dim=0)

        center_pos = torch.zeros((data["num_graphs"], 3), device=data["ligand_pos"].device)
        center_pos.index_add_(0, index=data["ligand_batch"], source=data["ligand_pos"])
        center_pos = center_pos / torch.bincount(data["ligand_batch"]).unsqueeze(1)

        edge_vec = data["ligand_pos"][edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand_node_sigma_emb"][edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], dim=1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Build graph for torsion angle prediction."""
        bonds = data["ligand_edge_index"][:, data["ligand_edge_mask"]].long()
        bond_pos = (data["ligand_pos"][bonds[0]] + data["ligand_pos"][bonds[1]]) / 2
        bond_batch = data["ligand_batch"][bonds[0]]
        edge_index = radius(
            data["ligand_pos"], bond_pos, self.lig_max_radius,
            batch_x=data["ligand_batch"], batch_y=bond_batch,
        )
        edge_vec = data["ligand_pos"][edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return bonds, edge_index, edge_attr, edge_sh, edge_weight

    def embedding(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        """Compute initial node and edge embeddings for all components."""
        # Receptor + atom embedding (cached across diffusion steps)
        if "rec_node_attr" not in data:
            rec_node_attr, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.build_rec_conv_graph(data)
            rec_node_attr = self.rec_node_embedding(rec_node_attr)
            rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

            atom_node_attr, atom_edge_attr, atom_edge_sh, atom_edge_weight = self.build_atom_conv_graph(data)
            atom_node_attr = self.atom_node_embedding(atom_node_attr)
            atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

            ar_edge_attr, ar_edge_sh, ar_edge_weight = self.build_cross_rec_conv_graph(data)
            ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

            rec_edge_index = data["receptor_edge_index"].clone()
            atom_edge_index = data["atom_edge_index"].clone()
            ar_edge_index = data["atom_receptor_edge_index"].clone()

            # Merge rec + atom into single graph for protein embedding layers
            node_attr = torch.cat([rec_node_attr, atom_node_attr], dim=0)
            ar_edge_index_shifted = ar_edge_index.clone()
            ar_edge_index_shifted[0] = ar_edge_index_shifted[0] + len(rec_node_attr)
            edge_index = torch.cat([
                rec_edge_index,
                ar_edge_index_shifted,
                atom_edge_index + len(rec_node_attr),
                torch.flip(ar_edge_index_shifted, dims=[0]),
            ], dim=1)
            edge_attr = torch.cat([rec_edge_attr, ar_edge_attr, atom_edge_attr, ar_edge_attr], dim=0)
            edge_sh = torch.cat([rec_edge_sh, ar_edge_sh, atom_edge_sh, ar_edge_sh], dim=0)
            edge_weight = torch.cat([rec_edge_weight, ar_edge_weight, atom_edge_weight, ar_edge_weight], dim=0) \
                if torch.is_tensor(rec_edge_weight) else torch.ones((edge_index.shape[1], 1), device=edge_index.device)
            s1 = len(rec_edge_index[0])
            s2 = s1 + len(ar_edge_index[0])
            s3 = s2 + len(atom_edge_index[0])

            for layer in self.rec_emb_layers:
                edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
                if self.differentiate_convolutions:
                    edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:]]
                node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight)

            # Cache results
            data["rec_node_attr"] = node_attr[:len(rec_node_attr)]
            data["rec_edge_attr"] = rec_edge_attr
            data["rec_edge_sh"] = rec_edge_sh
            data["rec_edge_weight"] = rec_edge_weight
            data["atom_node_attr"] = node_attr[len(rec_node_attr):]
            data["atom_edge_attr"] = atom_edge_attr
            data["atom_edge_sh"] = atom_edge_sh
            data["atom_edge_weight"] = atom_edge_weight
            data["ar_edge_attr"] = ar_edge_attr
            data["ar_edge_sh"] = ar_edge_sh
            data["ar_edge_weight"] = ar_edge_weight

        # Add timestep embedding
        rec_sigma_emb = self.rec_sigma_embedding(self.timestep_emb_func(data["complex_t"]["tr"]))
        rec_node_attr = data["rec_node_attr"].clone()
        rec_node_attr[:, :self.ns] += rec_sigma_emb[data["receptor_batch"]]
        rec_edge_attr = data["rec_edge_attr"] + rec_sigma_emb[data["receptor_batch"][data["receptor_edge_index"][0]]]

        atom_node_attr = data["atom_node_attr"].clone()
        atom_node_attr[:, :self.ns] += rec_sigma_emb[data["atom_batch"]]
        atom_edge_attr = data["atom_edge_attr"] + rec_sigma_emb[data["atom_batch"][data["atom_edge_index"][0]]]

        ar_edge_attr = data["ar_edge_attr"] + rec_sigma_emb[data["atom_batch"][data["atom_receptor_edge_index"][0]]]

        # Ligand embedding
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        if self.embed_also_ligand:
            for layer in self.lig_emb_layers:
                edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns],
                                        lig_node_attr[lig_edge_index[1], :self.ns]], -1)
                lig_node_attr = layer(lig_node_attr, lig_edge_index, edge_attr_, lig_edge_sh, edge_weight=lig_edge_weight)
        else:
            lig_node_attr = F.pad(lig_node_attr, (0, rec_node_attr.shape[-1] - lig_node_attr.shape[-1]))

        return (
            lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight,
            rec_node_attr, data["receptor_edge_index"], rec_edge_attr, data["rec_edge_sh"], data["rec_edge_weight"],
            atom_node_attr, data["atom_edge_index"], atom_edge_attr, data["atom_edge_sh"], data["atom_edge_weight"],
            data["atom_receptor_edge_index"], ar_edge_attr, data["ar_edge_sh"], data["ar_edge_weight"],
        )

    def forward(self, data: dict[str, Any]) -> tuple[Tensor, ...]:
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
                *[data["complex_t"][k] for k in ("tr", "rot", "tor")]
            )
        else:
            tr_sigma, rot_sigma, tor_sigma = [data["complex_t"][k] for k in ("tr", "rot", "tor")]

        (lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight,
         rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight,
         atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh, atom_edge_weight,
         ar_edge_index, ar_edge_attr, ar_edge_sh, ar_edge_weight) = self.embedding(data)

        # Build cross edges
        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        (lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight,
         la_edge_index, la_edge_attr, la_edge_sh, la_edge_weight) = self.build_cross_lig_conv_graph(data, cross_cutoff)
        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)

        # Merge all nodes and edges into a single heterogeneous graph
        n_lig, n_rec = len(lig_node_attr), len(rec_node_attr)
        node_attr = torch.cat([lig_node_attr, rec_node_attr, atom_node_attr], dim=0)

        # Shift edge indices to global node space
        rec_edge_index = rec_edge_index.clone()
        atom_edge_index = atom_edge_index.clone()
        lr_edge_index = lr_edge_index.clone()
        la_edge_index = la_edge_index.clone()
        ar_edge_index = ar_edge_index.clone()

        rec_edge_index += n_lig
        atom_edge_index += n_lig + n_rec
        lr_edge_index[1] += n_lig
        la_edge_index[1] += n_lig + n_rec
        ar_edge_index[0] += n_lig + n_rec
        ar_edge_index[1] += n_lig

        edge_index = torch.cat([
            lig_edge_index, lr_edge_index, la_edge_index, rec_edge_index,
            torch.flip(lr_edge_index, dims=[0]), torch.flip(ar_edge_index, dims=[0]),
            atom_edge_index, torch.flip(la_edge_index, dims=[0]), ar_edge_index,
        ], dim=1)
        edge_attr = torch.cat([
            lig_edge_attr, lr_edge_attr, la_edge_attr, rec_edge_attr, lr_edge_attr,
            ar_edge_attr, atom_edge_attr, la_edge_attr, ar_edge_attr,
        ], dim=0)
        edge_sh = torch.cat([
            lig_edge_sh, lr_edge_sh, la_edge_sh, rec_edge_sh, lr_edge_sh,
            ar_edge_sh, atom_edge_sh, la_edge_sh, ar_edge_sh,
        ], dim=0)
        edge_weight = torch.cat([
            lig_edge_weight, lr_edge_weight, la_edge_weight, rec_edge_weight, lr_edge_weight,
            ar_edge_weight, atom_edge_weight, la_edge_weight, ar_edge_weight,
        ], dim=0) if torch.is_tensor(lig_edge_weight) else torch.ones((edge_index.shape[1], 1), device=edge_index.device)

        lengths = [len(x) for x in [lig_edge_attr, lr_edge_attr, la_edge_attr, rec_edge_attr,
                                      lr_edge_attr, ar_edge_attr, atom_edge_attr, la_edge_attr, ar_edge_attr]]
        cumlen = list(np.cumsum(lengths))
        s1, s2, s3, s4, s5, s6, s7, s8 = cumlen[:-1]

        # Message passing
        for l_idx, layer in enumerate(self.conv_layers):
            if l_idx < len(self.conv_layers) - 1:
                edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
                if self.differentiate_convolutions:
                    edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:s4],
                                  edge_attr_[s4:s5], edge_attr_[s5:s6], edge_attr_[s6:s7], edge_attr_[s7:s8], edge_attr_[s8:]]
                node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight)
            else:
                # Last layer: only lig + cross edges (no rec-rec, atom-atom)
                edge_attr_ = torch.cat([edge_attr[:s3], node_attr[edge_index[0, :s3], :self.ns],
                                        node_attr[edge_index[1, :s3], :self.ns]], -1)
                if self.differentiate_convolutions:
                    edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3]]
                node_attr = layer(node_attr, edge_index[:, :s3], edge_attr_, edge_sh[:s3], edge_weight=edge_weight[:s3])

        lig_node_attr = node_attr[:n_lig]

        # Confidence prediction
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([
                lig_node_attr[:, :self.ns],
                lig_node_attr[:, -(self.nv if self.reduce_pseudoscalars else self.ns):],
            ], dim=1) if self.num_conv_layers + self.num_prot_emb_layers >= 3 else lig_node_attr[:, :self.ns]

            if self.atom_confidence:
                scalar_lig_attr = self.atom_confidence_predictor(scalar_lig_attr)
                atom_conf = scalar_lig_attr[:, :self.atom_num_confidence_outputs]
                scalar_lig_attr = scalar_lig_attr[:, self.atom_num_confidence_outputs:]
            else:
                atom_conf = torch.zeros(n_lig, device=lig_node_attr.device)

            confidence = self.confidence_predictor(
                scatter_mean(scalar_lig_attr, data["ligand_batch"], dim=0)
            ).squeeze(dim=-1)
            return confidence, atom_conf

        # Score prediction
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        else:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1)

        global_pred = self.final_conv(
            lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data["num_graphs"]
        )

        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

        if self.separate_noise_schedule:
            graph_sigma_emb = torch.cat([self.timestep_emb_func(data["complex_t"][k]) for k in ("tr", "rot", "tor")], dim=1)
        else:
            graph_sigma_emb = self.timestep_emb_func(data["complex_t"]["tr"])

        # Adjust score magnitudes
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, graph_sigma_emb], dim=1))

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            # Import so3 lazily to avoid circular imports
            from ...utils.so3 import score_norm as so3_score_norm
            rot_pred = rot_pred * so3_score_norm(rot_sigma.cpu()).unsqueeze(1).to(lig_node_attr.device)

        if self.no_torsion or data["ligand_edge_mask"].sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device), None

        # Torsion prediction
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = self.build_bond_conv_graph(data)
        tor_bond_vec = data["ligand_pos"][tor_bonds[1]] - data["ligand_pos"][tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization="component")
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([
            tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns], tor_bond_attr[tor_edge_index[0], :self.ns],
        ], -1)
        tor_pred = self.tor_bond_conv(
            lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
            out_nodes=data["ligand_edge_mask"].sum(), reduce="mean", edge_weight=tor_edge_weight,
        )
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)

        edge_sigma = tor_sigma[data["ligand_batch"]][data["ligand_edge_index"][0]][data["ligand_edge_mask"]]
        if self.scale_by_sigma:
            from ...utils.torus import score_norm as torus_score_norm
            tor_pred = tor_pred * torch.sqrt(
                torch.tensor(torus_score_norm(edge_sigma.cpu().numpy())).float().to(lig_node_attr.device)
            )

        return tr_pred, rot_pred, tor_pred, None
