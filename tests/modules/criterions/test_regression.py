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

from multimolecule.modules.criterions import Criterion, MSELoss
from multimolecule.modules.heads.config import HeadConfig


class TestMSELoss:
    """Test MSELoss (regression)."""

    @pytest.fixture
    def config(self):
        return HeadConfig(num_labels=1, problem_type="regression")

    @pytest.fixture
    def generic_criterion(self, config):
        return Criterion(config)

    @pytest.fixture
    def criterion(self, config):
        return MSELoss(config)

    def test_2d_input(self, generic_criterion, criterion):
        """Test 2D input: (batch_size, 1)"""
        logits = torch.randn(4, 1)
        labels = torch.randn(4, 1)

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_3d_input(self, generic_criterion, criterion):
        """Test 3D input: (batch_size, seq_len, 1)"""
        logits = torch.randn(2, 10, 1)
        labels = torch.randn(2, 10, 1)

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_dimension_mismatch(self, generic_criterion, criterion):
        """Test when logits have extra dimension: logits (2, 10, 1) vs labels (2, 10)"""
        logits = torch.randn(2, 10, 1)
        labels = torch.randn(2, 10)

        loss_generic = generic_criterion(logits, labels)
        loss_specialized = criterion(logits, labels)

        assert loss_generic is not None
        assert loss_specialized is not None
        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    def test_multi_output_regression(self):
        """Test regression with multiple outputs (num_labels > 1)"""
        config = HeadConfig(num_labels=3, problem_type="regression")
        criterion = MSELoss(config)

        logits = torch.randn(4, 3)
        labels = torch.randn(4, 3)

        loss = criterion(logits, labels)
        assert loss is not None

    def test_nested_tensor(self, criterion):
        """Test with NestedTensor input"""
        logits = NestedTensor([torch.randn(5, 1), torch.randn(3, 1)])
        labels = NestedTensor([torch.randn(5, 1), torch.randn(3, 1)])

        loss = criterion(logits, labels)
        assert loss is not None

    def test_custom_loss_params(self):
        """Test with custom loss parameters"""
        config = HeadConfig(num_labels=1, problem_type="regression", loss={"reduction": "sum"})
        criterion = MSELoss(config)

        logits = torch.randn(4, 1)
        labels = torch.randn(4, 1)

        loss = criterion(logits, labels)
        assert loss is not None

    def test_dtype_conversion(self):
        """Test that dtype conversion works correctly"""
        config = HeadConfig(num_labels=1, problem_type="regression")
        criterion = MSELoss(config)

        logits = torch.randn(4, 1).float()
        labels = torch.randn(4, 1).double()

        loss = criterion(logits, labels)
        assert loss is not None
        assert loss.dtype == torch.float32

    def test_gradient_flow(self, criterion):
        """Test that gradients flow correctly"""
        logits = torch.randn(4, 1, requires_grad=True)
        labels = torch.randn(4, 1)

        loss = criterion(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_zero_loss(self, criterion):
        """Test that identical inputs produce near-zero loss"""
        logits = torch.randn(4, 1)
        labels = logits.clone()

        loss = criterion(logits, labels)
        assert loss is not None
        assert loss.item() < 1e-6

    def test_positive_loss(self, criterion):
        """Test that different inputs produce positive loss"""
        logits = torch.randn(4, 1)
        labels = torch.randn(4, 1)

        loss = criterion(logits, labels)
        assert loss is not None
        assert loss.item() >= 0
