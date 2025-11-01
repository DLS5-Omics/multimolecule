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
from torch import Tensor

from multimolecule.models import PreTrainedConfig
from multimolecule.modules.heads import BasePredictionHead, PredictionHead
from multimolecule.modules.heads.config import HeadConfig


class TestBasePredictionHead:

    config = PreTrainedConfig(hidden_size=64, head=None)
    head = BasePredictionHead(config)

    def test_get_attention_mask_with_tensor(self):
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = self.head.get_attention_mask(input_ids)
        expected_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        assert torch.equal(attention_mask, expected_mask)

    def test_get_attention_mask_with_nested_tensor(self):
        nested_input = NestedTensor([torch.tensor([1, 3, 4, 5, 2]), torch.tensor([1, 3, 2])])
        attention_mask = self.head.get_attention_mask(nested_input)
        assert torch.equal(attention_mask, nested_input.mask)

    def test_get_attention_mask_none_input_ids(self):
        with pytest.raises(
            ValueError, match="Unable to infer attention mask for BasePredictionHead, because input_ids is None"
        ):
            self.head.get_attention_mask(None)

    def test_get_attention_mask_no_pad_token_id(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, pad_token_id=None))
        input_ids = torch.tensor([[1, 3, 4, 2, 0]])
        with pytest.raises(
            ValueError, match="Unable to infer attention mask for BasePredictionHead, because pad_token_id is None"
        ):
            head.get_attention_mask(input_ids)

    def test_remove_special_tokens_both_bos_eos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, hidden_size)

        new_output, new_attention_mask, _ = head.remove_special_tokens(output, attention_mask)
        assert torch.equal(new_output, NestedTensor.from_tensor_mask(output, attention_mask)[:, 1:-1, :].tensor)
        assert torch.equal(new_attention_mask, torch.tensor([[1, 1, 1], [1, 0, 0]]).int())

    def test_remove_special_tokens_only_bos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, eos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens(output, attention_mask, input_ids)

        assert torch.equal(new_output, NestedTensor.from_tensor_mask(output, attention_mask)[:, 1:, :].tensor)
        assert torch.equal(new_attention_mask, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).int())
        assert torch.equal(new_input_ids, torch.tensor([[3, 4, 5, 2], [3, 2, 0, 0]]))

    def test_remove_special_tokens_only_eos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, bos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens(output, attention_mask, input_ids)

        assert torch.equal(new_output, NestedTensor.from_tensor_mask(output, attention_mask)[:, :-1, :].tensor)
        assert torch.equal(new_attention_mask, torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).int())
        assert torch.equal(new_input_ids, torch.tensor([[1, 3, 4, 5], [1, 3, 0, 0]]))

    def test_remove_special_tokens_none(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, bos_token_id=None, eos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens(output, attention_mask, input_ids)

        assert torch.equal(new_output, NestedTensor.from_tensor_mask(output, attention_mask).tensor)
        assert torch.equal(new_attention_mask, attention_mask)
        assert torch.equal(new_input_ids, input_ids)

    def test_remove_special_tokens_2d_both_bos_eos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, seq_len, hidden_size)

        new_output, new_attention_mask, _ = head.remove_special_tokens_2d(output, attention_mask)

        true_attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]]).int()
        true_attention_mask = true_attention_mask.unsqueeze(1) * true_attention_mask.unsqueeze(2)
        true_output = output[:, 1:-1, 1:-1, :] * true_attention_mask.unsqueeze(-1)

        assert torch.equal(new_output, true_output)
        assert torch.equal(new_attention_mask, true_attention_mask)

    def test_remove_special_tokens_2d_only_bos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, eos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens_2d(output, attention_mask, input_ids)

        true_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).int()
        true_attention_mask = true_attention_mask.unsqueeze(1) * true_attention_mask.unsqueeze(2)
        true_output = output[:, 1:, 1:, :] * true_attention_mask.unsqueeze(-1)

        assert torch.equal(new_output, true_output)
        assert torch.equal(new_attention_mask, true_attention_mask)
        assert torch.equal(new_input_ids, torch.tensor([[3, 4, 5, 2], [3, 2, 0, 0]]))

    def test_remove_special_tokens_2d_only_eos(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, bos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens_2d(output, attention_mask, input_ids)

        true_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).int()
        true_attention_mask = true_attention_mask.unsqueeze(1) * true_attention_mask.unsqueeze(2)
        true_output = output[:, :-1, :-1, :] * true_attention_mask.unsqueeze(-1)

        assert torch.equal(new_output, true_output)
        assert torch.equal(new_attention_mask, true_attention_mask)
        assert torch.equal(new_input_ids, torch.tensor([[1, 3, 4, 5], [1, 3, 0, 0]]))

    def test_remove_special_tokens_2d_none(self):
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None, bos_token_id=None, eos_token_id=None))

        batch_size, seq_len, hidden_size = 2, 5, 4
        input_ids = torch.tensor([[1, 3, 4, 5, 2], [1, 3, 2, 0, 0]])
        attention_mask = head.get_attention_mask(input_ids)
        output = torch.randn(batch_size, seq_len, seq_len, hidden_size)

        new_output, new_attention_mask, new_input_ids = head.remove_special_tokens_2d(output, attention_mask, input_ids)

        true_attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        true_output = output * true_attention_mask.unsqueeze(-1)

        assert torch.equal(new_output, true_output)
        assert torch.equal(new_attention_mask, true_attention_mask)
        assert torch.equal(new_input_ids, input_ids)

    def test_head_config_inheritance(self):
        head_config = HeadConfig(num_labels=5, hidden_size=512)
        head = BasePredictionHead(PreTrainedConfig(hidden_size=64, head=None), head_config)

        assert head.num_labels == 5
        assert head.config.hidden_size == 512

    def test_edge_cases_attention_mask(self):
        # Test with all padding
        all_pad_ids = torch.zeros(2, 3, dtype=torch.long)
        attention_mask = self.head.get_attention_mask(all_pad_ids)
        expected = torch.zeros(2, 3, dtype=torch.int)
        assert torch.equal(attention_mask, expected)

        # Test with no padding
        no_pad_ids = torch.ones(2, 3, dtype=torch.long)
        attention_mask = self.head.get_attention_mask(no_pad_ids)
        expected = torch.ones(2, 3, dtype=torch.int)
        assert torch.equal(attention_mask, expected)


class TestPredictionHead:

    config = PreTrainedConfig(hidden_size=64, head=None)
    head_config = HeadConfig(num_labels=3, hidden_size=768, dropout=0.1, act="gelu", problem_type="multiclass")
    head = PredictionHead(config, head_config)

    @pytest.mark.parametrize(
        "labels",
        [
            None,
            torch.randint(0, 3, (2, 28)),
            NestedTensor([torch.randint(0, 3, (13,)), torch.randint(0, 3, (28,))], padding_value=-100),
        ],
    )
    def test_forward(self, labels):
        batch_size, seq_len, hidden_size = 2, 28, 768
        embeddings = torch.randn(batch_size, seq_len, hidden_size)
        if labels is not None:
            labels = labels.unsqueeze(-1)

        output = self.head(embeddings, labels=labels)

        assert isinstance(output.logits, (Tensor, NestedTensor))
        assert output.logits.shape == (batch_size, seq_len, self.head.num_labels)
        if labels is None:
            assert output.loss is None
        else:
            assert isinstance(output.loss, Tensor)
            assert output.loss.requires_grad
