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

"""SE(3)-equivariant tensor product convolution layers for DiffDock.

Replaces torch_scatter with native PyTorch scatter_reduce operations.
e3nn is kept as the core equivariant compute engine.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.o3 import Linear, TensorProduct
from torch import Tensor

from .graph_utils import scatter
from .layers import FCBlock


def get_irrep_seq(
    ns: int, nv: int, use_second_order_repr: bool, reduce_pseudoscalars: bool
) -> list[str]:
    """Get the sequence of irreducible representations for each conv layer depth."""
    if use_second_order_repr:
        return [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {nv if reduce_pseudoscalars else ns}x0o",
        ]
    return [
        f"{ns}x0e",
        f"{ns}x0e + {nv}x1o",
        f"{ns}x0e + {nv}x1o + {nv}x1e",
        f"{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o",
    ]


def irrep_to_size(irrep: str) -> int:
    """Compute the total feature dimension of an irrep string."""
    size = 0
    for ir in irrep.split(" + "):
        m, lp = ir.split("x")
        l_val = int(lp[0])
        size += int(m) * (2 * l_val + 1)
    return size


class FasterTensorProduct(nn.Module):
    """Optimized first-order tensor product (by Bowen Jing).

    Only supports sh_irreps = '1x0e + 1x1o' (first-order spherical harmonics).
    """

    def __init__(self, in_irreps: str, sh_irreps: str, out_irreps: str, **kwargs):
        super().__init__()
        assert o3.Irreps(sh_irreps) == o3.Irreps("1x0e+1x1o"), \
            "sh_irreps must be first order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        in_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        out_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        for m, ir in self.in_irreps:
            in_muls[str(ir)] = m
        for m, ir in self.out_irreps:
            out_muls[str(ir)] = m

        self.weight_shapes = {
            "0e": (in_muls["0e"] + in_muls["1o"], out_muls["0e"]),
            "1o": (in_muls["0e"] + in_muls["1o"] + in_muls["1e"], out_muls["1o"]),
            "1e": (in_muls["1o"] + in_muls["1e"] + in_muls["0o"], out_muls["1e"]),
            "0o": (in_muls["1e"] + in_muls["0o"], out_muls["0o"]),
        }
        self.weight_numel = sum(a * b for a, b in self.weight_shapes.values())

    def forward(self, in_: Tensor, sh: Tensor, weight: Tensor) -> Tensor:
        in_dict: dict[str, Tensor] = {}
        out_dict: dict[str, list[Tensor]] = {"0e": [], "1o": [], "1e": [], "0o": []}

        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]
            if ir[0] == 1:
                in_dict[str(ir)] = in_dict[str(ir)].reshape(list(in_dict[str(ir)].shape)[:-1] + [-1, 3])

        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]

        if "0e" in in_dict:
            out_dict["0e"].append(in_dict["0e"] * sh_0e.unsqueeze(-1))
            out_dict["1o"].append(in_dict["0e"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if "1o" in in_dict:
            out_dict["0e"].append((in_dict["1o"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
            out_dict["1o"].append(in_dict["1o"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["1e"].append(torch.linalg.cross(in_dict["1o"], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
        if "1e" in in_dict:
            out_dict["1o"].append(torch.linalg.cross(in_dict["1e"], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
            out_dict["1e"].append(in_dict["1e"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["0o"].append((in_dict["1e"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
        if "0o" in in_dict:
            out_dict["1e"].append(in_dict["0o"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict["0o"].append(in_dict["0o"] * sh_0e.unsqueeze(-1))

        weight_dict: dict[str, Tensor] = {}
        start = 0
        for key in self.weight_shapes:
            in_s, out_s = self.weight_shapes[key]
            weight_dict[key] = weight[..., start : start + in_s * out_s].reshape(
                list(weight.shape)[:-1] + [in_s, out_s]
            ) / np.sqrt(in_s)
            start += in_s * out_s

        if out_dict["0e"]:
            out_dict["0e"] = [torch.matmul(torch.cat(out_dict["0e"], dim=-1).unsqueeze(-2), weight_dict["0e"]).squeeze(-2)]
        if out_dict["1o"]:
            t = torch.cat(out_dict["1o"], dim=-2)
            t = (t.unsqueeze(-2) * weight_dict["1o"].unsqueeze(-1)).sum(-3)
            out_dict["1o"] = [t.reshape(list(t.shape)[:-2] + [-1])]
        if out_dict["1e"]:
            t = torch.cat(out_dict["1e"], dim=-2)
            t = (t.unsqueeze(-2) * weight_dict["1e"].unsqueeze(-1)).sum(-3)
            out_dict["1e"] = [t.reshape(list(t.shape)[:-2] + [-1])]
        if out_dict["0o"]:
            out_dict["0o"] = [torch.matmul(torch.cat(out_dict["0o"], dim=-1).unsqueeze(-2), weight_dict["0o"]).squeeze(-2)]

        out = []
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)][0] if out_dict[str(ir)] else torch.zeros(1))
        return torch.cat(out, dim=-1)


def tp_scatter_simple(
    tp: nn.Module,
    fc_layer: nn.Module,
    node_attr: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    edge_sh: Tensor,
    out_nodes: int | None = None,
    reduce: str = "mean",
    edge_weight: Tensor | float = 1.0,
) -> Tensor:
    """Tensor product + scatter for a single edge group."""
    _device = node_attr.device
    _dtype = node_attr.dtype
    edge_src, edge_dst = edge_index
    out_irreps = fc_layer(edge_attr).to(_device).to(_dtype)
    out_irreps.mul_(edge_weight)
    tp_out = tp(node_attr[edge_dst], edge_sh, out_irreps)
    out_nodes = out_nodes or node_attr.shape[0]
    return scatter(tp_out, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)


def tp_scatter_multigroup(
    tp: o3.TensorProduct,
    fc_layer: Union[nn.Module, nn.ModuleList],
    node_attr: Tensor,
    edge_index: Tensor,
    edge_attr_groups: List[Tensor],
    edge_sh: Tensor,
    out_nodes: int | None = None,
    reduce: str = "mean",
    edge_weight: Tensor | float = 1.0,
) -> Tensor:
    """Tensor product + scatter for multiple edge groups.

    Processes each group separately to keep peak memory low, then aggregates.
    """
    assert reduce in {"mean", "sum"}
    _device = node_attr.device
    _dtype = node_attr.dtype
    edge_src, edge_dst = edge_index
    edge_attr_lengths = [ea.shape[0] for ea in edge_attr_groups]
    num_edge_groups = len(edge_attr_groups)
    edge_weight_is_indexable = hasattr(edge_weight, "__getitem__")

    out_nodes = out_nodes or node_attr.shape[0]
    total_output_dim = sum(x.dim for x in tp.irreps_out)
    final_out = torch.zeros((out_nodes, total_output_dim), device=_device, dtype=_dtype)
    div_factors = torch.zeros(out_nodes, device=_device, dtype=_dtype)

    cur_start = 0
    for ii in range(num_edge_groups):
        cur_length = edge_attr_lengths[ii]
        cur_end = cur_start + cur_length
        cur_edge_range = slice(cur_start, cur_end)
        cur_edge_src, cur_edge_dst = edge_src[cur_edge_range], edge_dst[cur_edge_range]

        cur_fc = fc_layer[ii] if isinstance(fc_layer, nn.ModuleList) else fc_layer
        cur_out_irreps = cur_fc(edge_attr_groups[ii])
        if edge_weight_is_indexable:
            cur_out_irreps.mul_(edge_weight[cur_edge_range])
        else:
            cur_out_irreps.mul_(edge_weight)

        summand = tp(node_attr[cur_edge_dst, :], edge_sh[cur_edge_range, :], cur_out_irreps)
        final_out += scatter(summand, cur_edge_src, dim=0, dim_size=out_nodes, reduce="sum")
        div_factors += torch.bincount(cur_edge_src, minlength=out_nodes).float()

        cur_start = cur_end
        del cur_out_irreps, summand

    if reduce == "mean":
        div_factors = torch.clamp(div_factors, torch.finfo(_dtype).eps)
        final_out = final_out / div_factors[:, None]

    return final_out


class TensorProductConvLayer(nn.Module):
    """SE(3)-equivariant graph convolution via tensor products.

    This is the core message-passing layer of DiffDock, rewritten without PyG dependencies.
    It uses e3nn for equivariant tensor products and native PyTorch scatter for aggregation.
    """

    def __init__(
        self,
        in_irreps: str,
        sh_irreps: str,
        out_irreps: str,
        n_edge_features: int,
        residual: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.0,
        hidden_features: int | None = None,
        faster: bool = False,
        edge_groups: int = 1,
        tp_weights_layers: int = 2,
        activation: str = "relu",
        depthwise: bool = False,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.edge_groups = edge_groups
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features

        if depthwise:
            in_irreps_obj = o3.Irreps(in_irreps)
            sh_irreps_obj = o3.Irreps(sh_irreps)
            out_irreps_obj = o3.Irreps(out_irreps)

            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(in_irreps_obj):
                for j, (_, ir_edge) in enumerate(sh_irreps_obj):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in out_irreps_obj:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()
            instructions = [
                (i_in1, i_in2, p[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in instructions
            ]

            self.tp = TensorProduct(
                in_irreps_obj, sh_irreps_obj, irreps_mid,
                instructions, shared_weights=False, internal_weights=False,
            )
            self.linear_2 = Linear(
                irreps_in=irreps_mid.simplify(),
                irreps_out=out_irreps_obj,
                internal_weights=True,
                shared_weights=True,
            )
        else:
            if faster:
                self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
            else:
                self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        if edge_groups == 1:
            self.fc = FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
        else:
            self.fc = nn.ModuleList([
                FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
                for _ in range(edge_groups)
            ])

        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(
        self,
        node_attr: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | list[Tensor],
        edge_sh: Tensor,
        out_nodes: int | None = None,
        reduce: str = "mean",
        edge_weight: Tensor | float = 1.0,
    ) -> Tensor:
        if edge_index.shape[1] == 0 and node_attr.shape[0] == 0:
            raise ValueError("No edges and no nodes")

        _dtype = node_attr.dtype
        if edge_index.shape[1] == 0:
            out = torch.zeros((node_attr.shape[0], self.out_size), dtype=_dtype, device=node_attr.device)
        else:
            if self.edge_groups == 1:
                out = tp_scatter_simple(
                    self.tp, self.fc, node_attr, edge_index, edge_attr, edge_sh, out_nodes, reduce, edge_weight
                )
            else:
                out = tp_scatter_multigroup(
                    self.tp, self.fc, node_attr, edge_index, edge_attr, edge_sh, out_nodes, reduce, edge_weight
                )

            if self.depthwise:
                out = self.linear_2(out)

            if self.batch_norm:
                out = self.batch_norm(out)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out.to(_dtype)
