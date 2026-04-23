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


import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from multimolecule.utils import calculate_flops, calculate_macs


def _expected_lstm_ops(
    input_size: int,
    hidden_size: int,
    total_steps: int,
    *,
    num_layers: int = 1,
    bidirectional: bool = False,
    bias: bool = True,
    proj_size: int = 0,
):
    num_directions = 2 if bidirectional else 1
    recurrent_size = proj_size or hidden_size
    layer_input_size = input_size
    flops = 0
    macs = 0
    pointwise_flops_per_step = hidden_size * (3 * 4 + 2 * 5 + 4)
    bias_flops_per_step = 8 * hidden_size if bias else 0

    for _ in range(num_layers):
        for _ in range(num_directions):
            gate_macs = total_steps * 4 * hidden_size * (layer_input_size + recurrent_size)
            macs += gate_macs
            flops += 2 * gate_macs
            flops += total_steps * (bias_flops_per_step + pointwise_flops_per_step)

            if proj_size > 0:
                proj_macs = total_steps * hidden_size * proj_size
                macs += proj_macs
                flops += 2 * proj_macs

        layer_input_size = recurrent_size * num_directions

    return flops, macs


def test_calculate_ops_lstm_tensor_input():
    model = nn.LSTM(
        input_size=3,
        hidden_size=5,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
    )
    inputs = torch.randn(2, 4, 3)
    expected_flops, expected_macs = _expected_lstm_ops(
        input_size=3,
        hidden_size=5,
        total_steps=8,
        num_layers=2,
        bidirectional=True,
    )

    assert calculate_flops(model, inputs) == expected_flops
    assert calculate_macs(model, inputs) == expected_macs


def test_calculate_ops_lstm_subclass_with_packed_sequence():
    class PackedLstm(nn.LSTM):
        pass

    model = PackedLstm(input_size=3, hidden_size=5, batch_first=True)
    padded = torch.randn(3, 4, 3)
    lengths = torch.tensor([4, 2, 1])
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    expected_flops, expected_macs = _expected_lstm_ops(input_size=3, hidden_size=5, total_steps=7)

    assert calculate_flops(model, packed) == expected_flops
    assert calculate_macs(model, packed) == expected_macs


def test_calculate_ops_lstm_with_projection():
    model = nn.LSTM(input_size=4, hidden_size=6, proj_size=2, batch_first=True, bias=False)
    inputs = torch.randn(1, 3, 4)
    expected_flops, expected_macs = _expected_lstm_ops(
        input_size=4,
        hidden_size=6,
        total_steps=3,
        bias=False,
        proj_size=2,
    )

    assert calculate_flops(model, inputs) == expected_flops
    assert calculate_macs(model, inputs) == expected_macs
