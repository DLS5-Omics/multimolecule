# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

from __future__ import annotations

import tempfile

import torch
from danling import NestedTensor
from transformers import BertConfig

from multimolecule.modules.heads import HeadConfig
from multimolecule.modules.model import MultiMoleculeModel


class TestSequenceBackbone:
    def test_nested_tensor_sequence_to_contact_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            BertConfig(vocab_size=16, hidden_size=8).save_pretrained(tmp)
            model = MultiMoleculeModel(
                backbone={"type": "sequence", "sequence": {"type": "onehot", "pretrained": tmp}},
                heads={"contact": HeadConfig(type="contact.logits.linear", num_labels=1, problem_type="binary")},
            ).eval()
            sequence = NestedTensor(
                [
                    torch.tensor([1, 6, 7, 8, 9, 2]),
                    torch.tensor([1, 6, 7, 2]),
                ]
            )
            labels = NestedTensor([torch.zeros(6, 6, 1), torch.zeros(4, 4, 1)])

            output = model(sequence, contact=labels)
            backbone_output, _ = model.backbone(sequence)

        assert isinstance(backbone_output["last_hidden_state"], NestedTensor)
        assert isinstance(output["contact"].logits, NestedTensor)
        assert [logits.shape for logits in output["contact"].logits.unbind()] == [
            torch.Size([6, 6, 1]),
            torch.Size([4, 4, 1]),
        ]
        assert torch.isfinite(output["contact"].loss)
