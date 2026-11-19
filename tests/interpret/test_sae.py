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

from io import BytesIO

import pytest
import torch

from multimolecule.interpret.sae import SparseAutoencoder


def test_sae_reconstructs_with_normalized_decoder_and_reports_sparsity():
    sae = SparseAutoencoder(input_size=2, num_features=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(2))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(torch.diag(torch.tensor([2.0, 3.0])))
        sae.decoder.bias.zero_()

    activations = torch.tensor([[[2.0, -1.0], [-3.0, 4.0]]])
    output = sae(activations)
    expected = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    torch.testing.assert_close(output.features, expected)
    torch.testing.assert_close(output.reconstruction, expected)
    torch.testing.assert_close(sae.encode(activations), expected)
    torch.testing.assert_close(sae.decode(output.features), expected)

    loss = sae.loss(activations, sparsity_weight=0.25)
    torch.testing.assert_close(loss, torch.tensor(3.25))
    assert loss.requires_grad

    metrics = sae.evaluate(activations, sparsity_weight=0.25)
    torch.testing.assert_close(metrics.loss, loss.detach())
    torch.testing.assert_close(metrics.reconstruction_error, torch.tensor(2.5))
    torch.testing.assert_close(metrics.sparsity_penalty, torch.tensor(3.0))
    torch.testing.assert_close(metrics.mean_active_features, torch.tensor(1.0))
    assert not metrics.loss.requires_grad


@pytest.mark.parametrize("activations", [torch.tensor(2.0), torch.ones(2, 1)], ids=["scalar", "single-channel"])
def test_sae_rejects_activations_that_would_broadcast_to_expected_width(activations):
    sae = SparseAutoencoder(2, 3)

    with pytest.raises(ValueError, match="last dimension"):
        sae(activations)


def test_sae_trains_on_activations_and_round_trips_state_dict():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        sae = SparseAutoencoder(3, 5)
        restored = SparseAutoencoder(3, 5)

    activations = torch.tensor([[0.5, -1.0, 2.0], [-2.0, 0.0, 1.0]])
    optimizer = torch.optim.SGD(sae.parameters(), lr=0.01)
    optimizer.zero_grad()
    before = sae.loss(activations)
    before.backward()
    optimizer.step()
    assert sae.loss(activations).item() < before.item()

    checkpoint = BytesIO()
    torch.save(sae.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    reference = sae(activations)
    output = restored(activations)
    assert output.features.shape == (2, 5)
    assert output.reconstruction.shape == activations.shape
    torch.testing.assert_close(output.features, reference.features)
    torch.testing.assert_close(output.reconstruction, reference.reconstruction)
