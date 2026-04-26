# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import tempfile

import torch
from danling import NestedTensor
from transformers import BertConfig

from multimolecule.modules.backbones.sequences.onehot import OneHot


class TestOneHot:
    def test_forward_preserves_nested_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            BertConfig(vocab_size=16, hidden_size=8).save_pretrained(tmp)
            model = OneHot(tmp)
            input_ids = NestedTensor([torch.tensor([1, 6, 7]), torch.tensor([1, 6])])

            output = model(input_ids)

        assert isinstance(output["last_hidden_state"], NestedTensor)
        assert [hidden.shape for hidden in output["last_hidden_state"].unbind()] == [
            torch.Size([3, 8]),
            torch.Size([2, 8]),
        ]
        assert output["pooler_output"].shape == torch.Size([2, 8])
