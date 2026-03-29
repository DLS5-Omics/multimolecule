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

"""Molecular structure processing for DiffDock.

Converts RDKit molecules and PDB structures into dict-based graph representations.
Replaces PyG HeteroData with flat dicts and torch_cluster with graph_utils.

Required dependencies: rdkit, prody (install via pip install rdkit-pypi prody).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, cdist

from .graph_utils import knn_graph

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit.Geometry import Point3D
except ImportError:
    Chem = None  # type: ignore[assignment]

# Bond type mapping
_bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} if Chem is not None else {}

# Allowable feature vocabularies
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_numring_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring3_list": [False, True],
    "possible_is_in_ring4_list": [False, True],
    "possible_is_in_ring5_list": [False, True],
    "possible_is_in_ring6_list": [False, True],
    "possible_is_in_ring7_list": [False, True],
    "possible_is_in_ring8_list": [False, True],
    "possible_amino_acids": [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HIP", "HIE", "TPO", "HID", "LEV", "MEU", "PTR", "GLV", "CYT", "SEP",
        "HIZ", "CYM", "GLM", "ASQ", "TYS", "CYX", "GLZ", "misc",
    ],
    "possible_atom_type_2": [
        "C*", "CA", "CB", "CD", "CE", "CG", "CH", "CZ", "N*", "ND", "NE",
        "NH", "NZ", "O*", "OD", "OE", "OG", "OH", "OX", "S*", "SD", "SG", "misc",
    ],
    "possible_atom_type_3": [
        "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
        "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3", "N", "ND1", "ND2",
        "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O", "OD1", "OD2", "OE1",
        "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG", "misc",
    ],
}

# Feature dimension tuples (categorical_dims, num_scalar_features)
lig_feature_dims = (
    list(map(len, [
        allowable_features["possible_atomic_num_list"],
        allowable_features["possible_chirality_list"],
        allowable_features["possible_degree_list"],
        allowable_features["possible_formal_charge_list"],
        allowable_features["possible_implicit_valence_list"],
        allowable_features["possible_numH_list"],
        allowable_features["possible_number_radical_e_list"],
        allowable_features["possible_hybridization_list"],
        allowable_features["possible_is_aromatic_list"],
        allowable_features["possible_numring_list"],
        allowable_features["possible_is_in_ring3_list"],
        allowable_features["possible_is_in_ring4_list"],
        allowable_features["possible_is_in_ring5_list"],
        allowable_features["possible_is_in_ring6_list"],
        allowable_features["possible_is_in_ring7_list"],
        allowable_features["possible_is_in_ring8_list"],
    ])),
    0,
)

rec_residue_feature_dims = (list(map(len, [allowable_features["possible_amino_acids"]])), 0)

rec_atom_feature_dims = (
    list(map(len, [
        allowable_features["possible_amino_acids"],
        allowable_features["possible_atomic_num_list"],
        allowable_features["possible_atom_type_2"],
        allowable_features["possible_atom_type_3"],
    ])),
    0,
)


def safe_index(l: list, e: Any) -> int:
    """Return index of element e in list l, or last index if not found."""
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1


def lig_atom_featurizer(mol: Any) -> Tensor:
    """Extract atom-level features from an RDKit molecule.

    Returns:
        Tensor of shape (num_atoms, 16) with categorical feature indices.
    """
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        chiral_tag = str(atom.GetChiralTag())
        if chiral_tag in ["CHI_SQUAREPLANAR", "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL"]:
            chiral_tag = "CHI_OTHER"

        atom_features_list.append([
            safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
            allowable_features["possible_chirality_list"].index(str(chiral_tag)),
            safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
            safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
            safe_index(allowable_features["possible_implicit_valence_list"], atom.GetImplicitValence()),
            safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
            safe_index(allowable_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
            allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
            safe_index(allowable_features["possible_numring_list"], ringinfo.NumAtomRings(idx)),
            allowable_features["possible_is_in_ring3_list"].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features["possible_is_in_ring4_list"].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features["possible_is_in_ring5_list"].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features["possible_is_in_ring6_list"].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features["possible_is_in_ring7_list"].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features["possible_is_in_ring8_list"].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])
    return torch.tensor(atom_features_list)


def build_ligand_graph(mol: Any) -> dict[str, Tensor]:
    """Build a ligand graph from an RDKit molecule.

    Args:
        mol: RDKit molecule with 3D conformer.

    Returns:
        Dict with keys: ligand_x, ligand_pos, ligand_edge_index, ligand_edge_attr.
    """
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    x = lig_atom_featurizer(mol).float()

    # Build bond graph
    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        bt = _bonds.get(bond.GetBondType(), 0)
        edge_types += [bt, bt]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.zeros(len(edge_types), 4)
    for i, bt in enumerate(edge_types):
        edge_attr[i, bt] = 1.0

    return {
        "ligand_x": x,
        "ligand_pos": pos,
        "ligand_edge_index": edge_index,
        "ligand_edge_attr": edge_attr,
    }


def build_receptor_graph(
    ca_coords: Tensor,
    residue_features: Tensor,
    neighbor_cutoff: float = 20.0,
    max_neighbors: int | None = None,
    knn_only: bool = False,
) -> dict[str, Tensor]:
    """Build a receptor residue-level graph.

    Args:
        ca_coords: C-alpha coordinates, shape (N_res, 3).
        residue_features: Residue features, shape (N_res, F).
        neighbor_cutoff: Distance cutoff for edges.
        max_neighbors: Maximum neighbors per residue.
        knn_only: If True, use k-NN graph instead of distance cutoff.

    Returns:
        Dict with keys: receptor_x, receptor_pos, receptor_edge_index.
    """
    if knn_only:
        k = max_neighbors if max_neighbors else 32
        edge_index = knn_graph(ca_coords, k)
    else:
        distances = cdist(ca_coords, ca_coords)
        src_list, dst_list = [], []
        max_n = max_neighbors if max_neighbors else 1000
        for i in range(len(ca_coords)):
            dists_i = distances[i]
            dst = (dists_i < neighbor_cutoff).nonzero(as_tuple=True)[0].tolist()
            if i in dst:
                dst.remove(i)
            if len(dst) > max_n:
                dst = dists_i.argsort()[1 : max_n + 1].tolist()
            if len(dst) == 0:
                dst = [dists_i.argsort()[1].item()]
            src_list.extend([i] * len(dst))
            dst_list.extend(dst)
        edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)

    return {
        "receptor_x": residue_features,
        "receptor_pos": ca_coords,
        "receptor_edge_index": edge_index,
    }


def build_complex_graph(
    ligand_data: dict[str, Tensor],
    receptor_data: dict[str, Tensor],
    atom_data: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Merge ligand, receptor, and optionally atom graphs into a single complex dict.

    This is the dict-based replacement for PyG's HeteroData.

    Returns:
        Merged dict with all graph components and batch vectors initialized for single sample.
    """
    data: dict[str, Any] = {}
    data.update(ligand_data)
    data.update(receptor_data)

    if atom_data is not None:
        data.update(atom_data)

    # Initialize batch vectors for single sample (batch_size=1)
    data["ligand_batch"] = torch.zeros(len(data["ligand_x"]), dtype=torch.long)
    data["receptor_batch"] = torch.zeros(len(data["receptor_x"]), dtype=torch.long)
    if atom_data is not None and "atom_x" in data:
        data["atom_batch"] = torch.zeros(len(data["atom_x"]), dtype=torch.long)
    data["num_graphs"] = 1

    return data
