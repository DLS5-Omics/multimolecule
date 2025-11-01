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

from multimolecule.modules.criterions import Criterion, MultiLabelSoftMarginLoss
from multimolecule.modules.heads.config import HeadConfig


class TestMultiLabelSoftMarginLoss:
    """Test MultiLabelSoftMarginLoss (multilabel classification)."""

    @pytest.fixture
    def config(self):
        return HeadConfig(num_labels=5, problem_type="multilabel")

    @pytest.fixture
    def generic_criterion(self, config):
        return Criterion(config)

    @pytest.fixture
    def criterion(self, config):
        return MultiLabelSoftMarginLoss(config)

    def test_2d_input(self, generic_criterion, criterion):
        """Test 2D input: (batch_size, num_labels)"""
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4, 5)).float()

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_3d_input(self, generic_criterion, criterion):
        """Test 3D input: (batch_size, seq_len, num_labels)"""
        logits = torch.randn(2, 10, 5)
        labels = torch.randint(0, 2, (2, 10, 5)).float()

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_no_flattening_for_2d(self, criterion):
        """Test that 2D inputs are not flattened"""
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4, 5)).float()

        loss = criterion(logits, labels)
        assert loss is not None

    def test_flattening_for_3d(self, criterion):
        """Test that 3D inputs are flattened correctly"""
        logits = torch.randn(2, 10, 5)
        labels = torch.randint(0, 2, (2, 10, 5)).float()

        loss = criterion(logits, labels)
        assert loss is not None

    def test_nested_tensor(self, criterion):
        """Test with NestedTensor input"""
        logits = NestedTensor([torch.randn(5, 5), torch.randn(3, 5)])
        labels = NestedTensor([torch.randint(0, 2, (5, 5)).float(), torch.randint(0, 2, (3, 5)).float()])

        loss = criterion(logits, labels)
        assert loss is not None

    def test_custom_loss_params(self):
        """Test with custom loss parameters"""
        config = HeadConfig(num_labels=5, problem_type="multilabel", loss={"reduction": "sum"})
        criterion = MultiLabelSoftMarginLoss(config)

        logits = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4, 5)).float()

        loss = criterion(logits, labels)
        assert loss is not None

    def test_gradient_flow(self, criterion):
        """Test that gradients flow correctly"""
        logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.randint(0, 2, (4, 5)).float()

        loss = criterion(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_all_zeros(self, criterion):
        """Test with all zero labels"""
        logits = torch.randn(4, 5)
        labels = torch.zeros(4, 5)

        loss = criterion(logits, labels)
        assert loss is not None

    def test_all_ones(self, criterion):
        """Test with all one labels"""
        logits = torch.randn(4, 5)
        labels = torch.ones(4, 5)

        loss = criterion(logits, labels)
        assert loss is not None
