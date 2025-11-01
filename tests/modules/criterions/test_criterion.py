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

from multimolecule.modules.criterions import (
    BCEWithLogitsLoss,
    Criterion,
    CrossEntropyLoss,
    MSELoss,
    MultiLabelSoftMarginLoss,
)
from multimolecule.modules.heads.config import HeadConfig


class TestCriterion:
    """Test the generic Criterion class."""

    def test_auto_detect_regression(self):
        """Test auto-detection of regression problem type"""
        config = HeadConfig(num_labels=1, problem_type=None)
        criterion = Criterion(config)

        logits = torch.randn(4, 1)
        labels = torch.randn(4, 1)  # Floating point labels

        with pytest.warns(UserWarning, match=".*regression.*"):
            loss = criterion(logits, labels)

        assert loss is not None
        assert criterion.problem_type == "regression"

    def test_auto_detect_binary(self):
        """Test auto-detection of binary problem type"""
        config = HeadConfig(num_labels=1, problem_type=None)
        criterion = Criterion(config)

        logits = torch.randn(4, 1)
        labels = torch.randint(0, 2, (4, 1))  # Integer labels, num_labels=1

        with pytest.warns(UserWarning, match=".*binary.*"):
            loss = criterion(logits, labels)

        assert loss is not None
        assert criterion.problem_type == "binary"

    def test_auto_detect_multilabel(self):
        """Test auto-detection of multilabel problem type"""
        config = HeadConfig(num_labels=5, problem_type=None)
        criterion = Criterion(config)

        logits = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4, 5))  # 2 unique values, num_labels > 1

        with pytest.warns(UserWarning, match=".*multilabel.*"):
            loss = criterion(logits, labels)

        assert loss is not None
        assert criterion.problem_type == "multilabel"

    def test_auto_detect_multiclass(self):
        """Test auto-detection of multiclass problem type"""
        config = HeadConfig(num_labels=5, problem_type=None)
        criterion = Criterion(config)

        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))  # More than 2 unique values

        with pytest.warns(UserWarning, match=".*multiclass.*"):
            loss = criterion(logits, labels)

        assert loss is not None
        assert criterion.problem_type == "multiclass"

    def test_none_labels(self):
        """Test that None labels return None loss"""
        config = HeadConfig(num_labels=5, problem_type="multiclass")
        criterion = Criterion(config)

        logits = torch.randn(4, 5)
        loss = criterion(logits, None)

        assert loss is None

    def test_invalid_problem_type(self):
        """Test that invalid problem_type raises ValueError"""
        config = HeadConfig(num_labels=5, problem_type="invalid_type")
        criterion = Criterion(config)

        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        with pytest.raises(ValueError, match="problem_type should be one of"):
            criterion(logits, labels)


class TestCriterionConsistency:
    """Test consistency between generic and specialized criterions."""

    @pytest.mark.parametrize("batch_size,seq_len,num_labels", [(4, 1, 1), (2, 10, 1), (8, 5, 1)])
    def test_binary_consistency(self, batch_size, seq_len, num_labels):
        """Test that generic and specialized binary criterions produce the same results"""
        config = HeadConfig(num_labels=num_labels, problem_type="binary")
        generic = Criterion(config)
        specialized = BCEWithLogitsLoss(config)

        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, 2, (batch_size, seq_len, num_labels)).float()

        loss_generic = generic(logits, labels)
        loss_specialized = specialized(logits, labels)

        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size,seq_len,num_labels", [(4, 1, 5), (2, 10, 3), (8, 5, 7)])
    def test_multiclass_consistency(self, batch_size, seq_len, num_labels):
        """Test that generic and specialized multiclass criterions produce the same results"""
        config = HeadConfig(num_labels=num_labels, problem_type="multiclass")
        generic = Criterion(config)
        specialized = CrossEntropyLoss(config)

        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, num_labels, (batch_size, seq_len))

        loss_generic = generic(logits, labels)
        loss_specialized = specialized(logits, labels)

        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size,seq_len,num_labels", [(4, 1, 5), (2, 10, 3), (8, 5, 7)])
    def test_multilabel_consistency(self, batch_size, seq_len, num_labels):
        """Test that generic and specialized multilabel criterions produce the same results"""
        config = HeadConfig(num_labels=num_labels, problem_type="multilabel")
        generic = Criterion(config)
        specialized = MultiLabelSoftMarginLoss(config)

        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randint(0, 2, (batch_size, seq_len, num_labels)).float()

        loss_generic = generic(logits, labels)
        loss_specialized = specialized(logits, labels)

        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size,seq_len,num_labels", [(4, 1, 1), (2, 10, 1), (8, 5, 3)])
    def test_regression_consistency(self, batch_size, seq_len, num_labels):
        """Test that generic and specialized regression criterions produce the same results"""
        config = HeadConfig(num_labels=num_labels, problem_type="regression")
        generic = Criterion(config)
        specialized = MSELoss(config)

        logits = torch.randn(batch_size, seq_len, num_labels)
        labels = torch.randn(batch_size, seq_len, num_labels)

        loss_generic = generic(logits, labels)
        loss_specialized = specialized(logits, labels)

        torch.testing.assert_close(loss_generic, loss_specialized, rtol=1e-4, atol=1e-4)
