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
from typing import Any, Tuple

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_a2zchromatin import A2zChromatinConfig


class A2zChromatinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = A2zChromatinConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["A2zChromatinEncoder"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    init.kaiming_normal_(parameter, mode="fan_in", nonlinearity="relu")
                elif "bias" in name:
                    init.zeros_(parameter)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class A2zChromatinModel(A2zChromatinPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import A2zChromatinConfig, A2zChromatinModel, DnaTokenizer
        >>> config = A2zChromatinConfig()
        >>> model = A2zChromatinModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/a2zchromatin")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 925])
    """

    def __init__(self, config: A2zChromatinConfig):
        super().__init__(config)
        self.embeddings = A2zChromatinEmbedding(config)
        self.encoder = A2zChromatinEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> A2zChromatinModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        # The a2z-chromatin encoder collapses the sequence dimension through the bidirectional LSTM and a dense
        # projection, so the final feature vector is both the model's last hidden state and its pooled representation.
        sequence_output = self.encoder(embedding_output)

        return A2zChromatinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
        )


class A2zChromatinForSequencePrediction(A2zChromatinPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import A2zChromatinConfig, A2zChromatinForSequencePrediction, DnaTokenizer
        >>> config = A2zChromatinConfig()
        >>> model = A2zChromatinForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/a2zchromatin")
        >>> input = tokenizer(["ACGT" * 150, "TGCA" * 150], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, 1)))
        >>> output["logits"].shape
        torch.Size([2, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<...>)
    """

    def __init__(self, config: A2zChromatinConfig):
        super().__init__(config)
        self.model = A2zChromatinModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"chromatin_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"chromatin_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def postprocess(self, outputs: Any) -> Tensor:
        return torch.sigmoid(outputs["logits"])


class A2zChromatinEmbedding(nn.Module):
    """One-hot embedding layer for a2z-chromatin.

    a2z-chromatin does not use learned word embeddings; it consumes one-hot tokenizer channels transposed into
    `(batch_size, vocab_size, sequence_length)` for the 1D convolution stack. Converted checkpoints expand the upstream
    four nucleotide channels to the DNA IUPAC tokenizer alphabet through the first convolution weights.
    """

    def __init__(self, config: A2zChromatinConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            self._check_sequence_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_sequence_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        return inputs_embeds

    def _check_sequence_length(self, sequence_length: int):
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"a2z-chromatin expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class A2zChromatinEncoder(nn.Module):
    """DanQ-style 1D-CNN + bidirectional LSTM encoder.

    The upstream `build_DanQ` topology is:
    ``Conv1D(320, 26, activation=relu) -> Dropout(0.2) -> MaxPool1D(13, 13) -> Bidirectional(LSTM(320))
    -> Dropout(0.5) -> Dense(925)``.
    Keras `LSTM(units)` defaults to `return_sequences=False`, so the bidirectional layer emits only the
    concatenation of the final forward hidden state and the final backward hidden state (shape
    `(2 * lstm_hidden_size,)`). The final ``Dense(1, sigmoid)`` cell is exposed through the shared
    [`SequencePredictionHead`][multimolecule.SequencePredictionHead] in
    [`A2zChromatinForSequencePrediction`][multimolecule.models.A2zChromatinForSequencePrediction].
    """

    def __init__(self, config: A2zChromatinConfig):
        super().__init__()
        # Upstream Keras `Conv1D(activation='relu')` performs the convolution then applies the activation; no
        # padding (Keras default `valid`) and no batch normalization, matching the DanQ recipe.
        self.conv = nn.Conv1d(config.vocab_size, config.conv_channels, kernel_size=config.conv_kernel_size)
        self.activation = ACT2FN[config.hidden_act]
        self.conv_dropout = nn.Dropout(config.conv_dropout)
        # Keras `MaxPool1D(pool_size, strides)` with explicit strides; default padding is `valid` (floor-mode).
        self.pool = nn.MaxPool1d(kernel_size=config.pool_size, stride=config.pool_size)
        self.lstm = nn.LSTM(
            input_size=config.conv_channels,
            hidden_size=config.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        # The final dense projection consumes the concatenation of the final forward and backward LSTM hidden
        # states (size `2 * lstm_hidden_size`); the pooled sequence length only affects the LSTM scan range.
        self.dense = nn.Linear(2 * config.lstm_hidden_size, config.fc_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        # Convolution branch operates on (batch, channels, length).
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv_dropout(hidden_state)
        hidden_state = self.pool(hidden_state)
        # Switch to (batch, length, channels) for the LSTM. The default `return_sequences=False` Keras behaviour
        # corresponds to taking the final time-step's forward output (last position) and the final time-step's
        # backward output (first position), then concatenating them. PyTorch returns those directly through the
        # `(h_n)` tensor of shape `(num_layers * num_directions, batch, hidden)`.
        hidden_state = hidden_state.transpose(1, 2).contiguous()
        _, (h_n, _) = self.lstm(hidden_state)
        # h_n is (2, batch, hidden); concatenate forward (index 0) and backward (index 1) hidden states.
        hidden_state = torch.cat((h_n[0], h_n[1]), dim=-1)
        hidden_state = self.lstm_dropout(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state


@dataclass
class A2zChromatinModelOutput(ModelOutput):
    """
    Base class for outputs of the a2z-chromatin backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Sequence-level representation produced by the DanQ CNN+BLSTM encoder and dense projection. The upstream
            Keras model returns only this final feature vector rather than per-position hidden states.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Alias of `last_hidden_state`; this is the tensor consumed by
            [`SequencePredictionHead`][multimolecule.modules.SequencePredictionHead].
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Always `None`; a2z-chromatin does not record intermediate hidden states.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Always `None`; a2z-chromatin is a convolutional/recurrent model without attention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
