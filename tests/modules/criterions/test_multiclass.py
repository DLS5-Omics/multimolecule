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


import pytest
import torch
from danling import NestedTensor

from multimolecule.modules.criterions import Criterion, CrossEntropyLoss
from multimolecule.modules.heads.config import HeadConfig


class TestCrossEntropyLoss:
    """Test CrossEntropyLoss (multiclass classification)."""

    @pytest.fixture
    def config(self):
        return HeadConfig(num_labels=5, problem_type="multiclass")

    @pytest.fixture
    def generic_criterion(self, config):
        return Criterion(config)

    @pytest.fixture
    def criterion(self, config):
        return CrossEntropyLoss(config)

    def test_2d_input(self, generic_criterion, criterion):
        """Test 2D input: (batch_size, num_classes)"""
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_3d_input(self, generic_criterion, criterion):
        """Test 3D input: (batch_size, seq_len, num_classes)"""
        logits = torch.randn(2, 10, 5)
        labels = torch.randint(0, 5, (2, 10))

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_flattening_consistency(self, criterion):
        """Test that flattening happens correctly for all input dimensions"""
        # 2D case
        logits_2d = torch.randn(4, 5)
        labels_2d = torch.randint(0, 5, (4,))
        loss_2d = criterion(logits_2d, labels_2d)

        # 3D case that should flatten to the same shape
        logits_3d = logits_2d.view(2, 2, 5)
        labels_3d = labels_2d.view(2, 2)
        loss_3d = criterion(logits_3d, labels_3d)

        assert loss_2d is not None
        assert loss_3d is not None
        torch.testing.assert_close(loss_2d, loss_3d, rtol=1e-4, atol=1e-4)

    def test_nested_tensor(self, criterion):
        """Test with NestedTensor input"""
        logits = NestedTensor([torch.randn(5, 5), torch.randn(3, 5)])
        labels = NestedTensor([torch.randint(0, 5, (5,)), torch.randint(0, 5, (3,))])

        loss = criterion(logits, labels)
        assert loss is not None

    def test_custom_loss_params(self):
        """Test with custom loss parameters"""
        config = HeadConfig(num_labels=5, problem_type="multiclass", loss={"ignore_index": -100})
        criterion = CrossEntropyLoss(config)

        logits = torch.randn(4, 10, 100)
        labels = torch.randint(100, (4, 10)).float()
        labels[-1, 5:] = -100

        loss = criterion(logits, labels)
        assert loss is not None

    def test_empty_batch(self):
        """Test with empty batch"""
        config = HeadConfig(num_labels=5, problem_type="multiclass")
        criterion = CrossEntropyLoss(config)

        logits = torch.randn(0, 5)
        labels = torch.randint(0, 5, (0,))

        loss = criterion(logits, labels)
        assert loss is not None
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_single_sample(self):
        """Test with single sample"""
        config = HeadConfig(num_labels=5, problem_type="multiclass")
        criterion = CrossEntropyLoss(config)

        logits = torch.randn(1, 5)
        labels = torch.tensor([2])

        loss = criterion(logits, labels)
        assert loss is not None

    def test_large_batch(self):
        """Test with large batch"""
        config = HeadConfig(num_labels=10, problem_type="multiclass")
        criterion = CrossEntropyLoss(config)

        logits = torch.randn(1000, 100, 10)
        labels = torch.randint(0, 10, (1000, 100))

        loss = criterion(logits, labels)
        assert loss is not None

    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        config = HeadConfig(num_labels=5, problem_type="multiclass")
        criterion = CrossEntropyLoss(config)

        logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.randint(0, 5, (4,))

        loss = criterion(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)
