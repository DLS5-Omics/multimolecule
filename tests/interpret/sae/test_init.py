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


class TestSparseAutoencoder:
    @pytest.fixture
    def sae(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            return SparseAutoencoder(3, 5)

    @pytest.fixture
    def activations(self):
        return torch.tensor([[0.5, -1.0, 2.0], [-2.0, 0.0, 1.0]])

    def test_encodes_nonnegative_features(self, sae, activations):
        features = sae.encode(activations.unsqueeze(0))

        assert features.shape == (1, 2, 5)
        assert torch.all(features >= 0)

    def test_decoder_normalizes_feature_contributions(self, sae):
        baseline = sae.decode(torch.zeros(5))
        contributions = sae.decode(torch.eye(5)) - baseline

        torch.testing.assert_close(contributions.norm(dim=-1), torch.ones(5))

    def test_reconstructs_each_activation(self, sae, activations):
        activations = activations.unsqueeze(0)
        output = sae(activations)

        assert output.reconstruction.shape == activations.shape
        torch.testing.assert_close(output.features, sae.encode(activations))
        torch.testing.assert_close(output.reconstruction, sae.decode(output.features))

    def test_reports_objective_terms(self, sae, activations):
        output = sae(activations)
        metrics = sae.evaluate(activations, sparsity_weight=0.25)
        error = (output.reconstruction - activations).square().mean()
        penalty = output.features.abs().sum(dim=-1).mean()

        torch.testing.assert_close(metrics.reconstruction_error, error)
        torch.testing.assert_close(metrics.sparsity_penalty, penalty)
        torch.testing.assert_close(metrics.mean_active_features, (output.features > 0).float().sum(dim=-1).mean())
        torch.testing.assert_close(metrics.loss, error + 0.25 * penalty)

    def test_loss_weights_sparsity(self, sae, activations):
        loss = sae.loss(activations, sparsity_weight=0.25)

        torch.testing.assert_close(loss, sae.evaluate(activations, sparsity_weight=0.25).loss)

    def test_evaluation_omits_gradients(self, sae, activations):
        assert not sae.evaluate(activations).loss.requires_grad

    @pytest.mark.parametrize("invalid", [torch.tensor(2.0), torch.ones(2, 1)], ids=["scalar", "single-channel"])
    def test_requires_input_width(self, sae, invalid):
        with pytest.raises(ValueError, match="last dimension"):
            sae(invalid)

    def test_optimizer_reduces_loss(self, sae, activations):
        optimizer = torch.optim.SGD(sae.parameters(), lr=0.01)
        before = sae.loss(activations)
        before.backward()
        optimizer.step()

        assert sae.loss(activations).item() < before.item()

    def test_round_trips_weights(self, sae, activations):
        checkpoint = BytesIO()
        torch.save(sae.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = SparseAutoencoder(3, 5)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))

        reference = sae(activations)
        output = restored(activations)
        torch.testing.assert_close(output.features, reference.features)
        torch.testing.assert_close(output.reconstruction, reference.reconstruction)
