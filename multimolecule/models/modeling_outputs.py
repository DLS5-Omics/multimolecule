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


from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class SequencePredictorOutput(ModelOutput):
    """
    Base class for outputs of sentence classification & regression models.

    Args:
        loss:
            `torch.FloatTensor` of shape `(1,)`.

            Optional, returned when `labels` is provided
        logits:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`

            Prediction outputs.
        hidden_states:
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Optional, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Optional, eturned when `output_attentions=True` is passed or when `config.output_attentions=True`

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class TokenPredictorOutput(ModelOutput):
    """
    Base class for outputs of token classification & regression models.

    Args:
        loss:
            `torch.FloatTensor` of shape `(1,)`.

            Optional, returned when `labels` is provided
        logits:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`

            Prediction outputs.
        hidden_states:
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Optional, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Optional, eturned when `output_attentions=True` is passed or when `config.output_attentions=True`

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ContactPredictorOutput(ModelOutput):
    """
    Base class for outputs of contact classification & regression models.

    Args:
        loss:
            `torch.FloatTensor` of shape `(1,)`.

            Optional, returned when `labels` is provided
        logits:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`

            Prediction outputs.
        hidden_states:
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Optional, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions:
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Optional, eturned when `output_attentions=True` is passed or when `config.output_attentions=True`

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
