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

import math
from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import HeadConfig, HeadOutput

from ..modeling_outputs import SequencePredictorOutput
from .configuration_deepmel import DeepMelConfig


class DeepMelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepMelConfig
    base_model_prefix = "model"
    # LSTM state is not trivially recomputable (hard-sigmoid custom cell + variational dropout), so gradient
    # checkpointing cannot be safely enabled without a validated custom recompute strategy.
    supports_gradient_checkpointing = False
    _can_record_outputs: dict[str, Any] | None = None
    # The whole encoder is indivisible: the LSTM carries recurrent state across Conv1D → TimeDistributed →
    # BiLSTM → FC, so splitting at a sub-layer boundary would break the recurrent forward pass.
    _no_split_modules = ["DeepMelEncoder"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, DeepMelLstm):
            init.xavier_uniform_(module.weight_ih)
            init.orthogonal_(module.weight_hh)
            init.zeros_(module.bias)


class DeepMelModel(DeepMelPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import DeepMelConfig, DeepMelModel, DnaTokenizer
        >>> config = DeepMelConfig()
        >>> model = DeepMelModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepmel")
        >>> input = tokenizer(["ACGT" * 125, "TGCA" * 125], return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([2, 256])
    """

    def __init__(self, config: DeepMelConfig):
        super().__init__(config)
        self.embeddings = DeepMelEmbedding(config)
        # Both the forward and reverse-complement branches share the same encoder weights, matching the upstream
        # Keras "siamese" model where `conv1d_1`, `time_distributed_1`, `bidirectional_1` and `dense_2` are each
        # applied to both `input_1` and `input_2` before the final averaging step.
        self.encoder = DeepMelEncoder(config)
        self.pooler = DeepMelPooler()

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
    ) -> DeepMelModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if isinstance(input_ids, NestedTensor):
            attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        forward_embedding = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        reverse_embedding = self.embeddings.reverse_complement(forward_embedding)

        forward_features = self.encoder(forward_embedding)
        reverse_features = self.encoder(reverse_embedding)
        # The forward and reverse-complement 256-dim FC representations are exposed in the output dataclass so that
        # `DeepMelForSequencePrediction` can run the final 24-way decoder on each branch and average the
        # post-sigmoid topic probabilities, matching the upstream Keras model exactly. The `pooler_output` is the
        # average of the two branches, which is the natural sequence-level embedding for backbone use cases.
        pooled_output = self.pooler(forward_features, reverse_features)

        return DeepMelModelOutput(
            last_hidden_state=forward_features,
            pooler_output=pooled_output,
            forward_hidden_state=forward_features,
            reverse_hidden_state=reverse_features,
        )


class DeepMelForSequencePrediction(DeepMelPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import DeepMelConfig, DeepMelForSequencePrediction, DnaTokenizer
        >>> config = DeepMelConfig()
        >>> model = DeepMelForSequencePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepmel")
        >>> input = tokenizer(["ACGT" * 125, "TGCA" * 125], return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (2, 24)).float())
        >>> output["logits"].shape
        torch.Size([2, 24])
    """

    def __init__(self, config: DeepMelConfig):
        super().__init__(config)
        self.model = DeepMelModel(config)
        # The upstream Keras model places the final 24-way Dense (`dense_3`) *inside each branch*, follows it with
        # a sigmoid, and then averages the two branches' probabilities. Because sigmoid is non-linear, that ordering
        # cannot be reproduced by the standard `SequencePredictionHead` (which averages its 256-dim input *before*
        # the decoder); we therefore own the decoder and expose the averaged probability in logit space.
        self.sequence_head = DeepMelSequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"topic_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        return [f"topic_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
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


class DeepMelSequencePredictionHead(nn.Module):
    """Per-branch decoder + sigmoid + averaging head.

    Mirrors the upstream Keras topology where each branch runs through `dense_3 -> sigmoid` and the two probability
    vectors are averaged. The model output stores `logit(mean(sigmoid(branch_logits)))` so `logits` keeps the usual
    MultiMolecule meaning while `postprocess` returns the upstream branch-averaged probability.
    """

    def __init__(self, config: DeepMelConfig):
        super().__init__()
        if config.head is None:
            raise ValueError("DeepMelConfig.head must be set; the default constructor builds a multilabel HeadConfig.")
        # Reuse the standard `HeadConfig` so `problem_type`, `num_labels` and the downstream pipeline metadata flow
        # through unchanged; only the actual decoder + loss live here.
        head_config: HeadConfig = config.head
        head_config.num_labels = config.num_labels
        head_config.hidden_size = config.fc_dim
        self.config = head_config
        self.num_labels = config.num_labels
        self.decoder = nn.Linear(config.fc_dim, config.num_labels, bias=True)
        self.dropout = nn.Dropout(config.fc_dropout)

    def forward(self, outputs: DeepMelModelOutput, labels: Tensor | None = None) -> HeadOutput:
        """Compute per-branch logits, average probabilities, re-logit, and optionally compute loss.

        Returns:
            HeadOutput where ``logits`` is ``logit(mean(sigmoid(branch_logits)))``, i.e. the averaged
            probability expressed back in logit space.  This value is BCE-with-logits compatible:
            ``torch.sigmoid(logits)`` recovers the upstream branch-averaged probability directly.

        Note on loss computation:
            The shared ``Criterion`` / multilabel path in ``SequencePredictionHead`` operates on raw
            logits (pre-sigmoid), but DeepMEL must average probabilities *between* the two branch
            sigmoids and *before* computing the loss.  Using ``sequence_head.criterion(logits, ...)``
            after re-logiting would introduce a double-sigmoid error, so the loss is computed directly
            on the averaged probability tensor with ``F.binary_cross_entropy``.
        """
        forward_features = outputs["forward_hidden_state"]
        reverse_features = outputs["reverse_hidden_state"]
        forward_probs = torch.sigmoid(self.decoder(self.dropout(forward_features)))
        reverse_probs = torch.sigmoid(self.decoder(self.dropout(reverse_features)))
        probabilities = (forward_probs + reverse_probs) / 2.0
        probabilities = probabilities.clamp(min=1e-7, max=1.0 - 1e-7)
        logits = torch.logit(probabilities)
        loss: Tensor | None = None
        if labels is not None:
            # BCE on the averaged probability (not on re-logited logits): the shared Criterion expects
            # raw logits and would apply sigmoid again, producing incorrect gradients for this model.
            loss = F.binary_cross_entropy(
                probabilities,
                labels.to(probabilities.dtype),
            )
        return HeadOutput(logits, loss)


class DeepMelEmbedding(nn.Module):
    """One-hot embedding layer for DeepMEL.

    DeepMEL does not use learned word embeddings; it consumes a one-hot encoding of the DNA nucleotides and
    keeps the `(batch_size, sequence_length, vocab_size)` layout used by the Keras Conv1D layer. The
    `reverse_complement` helper produces the matching reverse-complement embedding consumed by the second branch.
    """

    def __init__(self, config: DeepMelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.input_length = config.input_length
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16) so F.one_hot output
        # (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)
        # Reverse-complement permutation for the MultiMolecule DNA alphabet (`ACGTN...`). The first four channels
        # represent A, C, G, T and are swapped pairwise (A<->T, C<->G); any additional channels (e.g. `N` or IUPAC
        # symbols) are kept in place. Rebuilt in `forward` for transformers v5 meta-init safety.
        self._rc_indices: Tensor | None = None

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be specified when inputs_embeds is not provided")
            self._check_input_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_input_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return inputs_embeds

    def reverse_complement(self, embeddings: Tensor) -> Tensor:
        indices = self._reverse_complement_indices(embeddings.device)
        return embeddings.index_select(-1, indices).flip(-2)

    def _reverse_complement_indices(self, device: torch.device) -> Tensor:
        if self._rc_indices is None or self._rc_indices.device != device:
            order = list(range(self.vocab_size))
            # Swap A<->T and C<->G in the first four channels (MultiMolecule DNA alphabet starts with `ACGT`).
            if self.vocab_size >= 4:
                order[0], order[3] = order[3], order[0]
                order[1], order[2] = order[2], order[1]
            self._rc_indices = torch.tensor(order, dtype=torch.long, device=device)
        return self._rc_indices

    def _check_input_length(self, input_length: int):
        if input_length != self.input_length:
            raise ValueError(
                f"DeepMEL expects fixed-length {self.input_length} bp inputs, but got {input_length}. "
                "Pad or crop the sequence to match the configured input_length."
            )


class DeepMelEncoder(nn.Module):
    """Shared encoder applied to both the forward and reverse-complement branches.

    Mirrors the upstream Keras pipeline `Conv1D -> MaxPool -> Dropout -> TimeDistributed(Dense) ->
    Bidirectional(LSTM) -> Dropout -> Flatten -> Dense`, with the final per-branch dropout placed in
    `DeepMelSequencePredictionHead` (so it sits between the FC feature and the 24-way decoder, as in upstream).
    """

    def __init__(self, config: DeepMelConfig):
        super().__init__()
        self.input_length = config.input_length
        self.pool_size = config.pool_size
        self.pooled_length = config.pooled_length
        self.lstm_hidden_size = config.lstm_hidden_size
        self.conv = nn.Conv1d(
            in_channels=config.vocab_size,
            out_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            padding=0,
        )
        self.act = ACT2FN[config.hidden_act]
        self.pool = nn.MaxPool1d(kernel_size=config.pool_size, stride=config.pool_size)
        self.conv_dropout = nn.Dropout(config.conv_dropout)
        # The Keras `TimeDistributed(Dense)` and `Bidirectional(LSTM)` natively consume `(batch, time, features)`
        # tensors, so we work in time-major channel-last layout between the convolution and the FC layer.
        self.time_distributed = nn.Linear(config.conv_channels, config.time_distributed_channels)
        self.time_distributed_act = ACT2FN[config.hidden_act]
        # Upstream Keras 2.1.5 uses LSTM `implementation=1` with `recurrent_activation="hard_sigmoid"`, which differs
        # numerically from `torch.nn.LSTM` (sigmoid recurrent activation). A small custom cell reproduces the upstream
        # gate equations exactly so the converted checkpoint matches the original within float32 noise.
        self.lstm = DeepMelBidirectionalLstm(
            input_size=config.time_distributed_channels,
            hidden_size=config.lstm_hidden_size,
            dropout=config.lstm_dropout,
            recurrent_dropout=config.lstm_recurrent_dropout,
        )
        self.recurrent_dropout = nn.Dropout(config.recurrent_dropout)
        self.fc = nn.Linear(config.fc_input_size, config.fc_dim)
        self.fc_act = ACT2FN[config.hidden_act]

    def forward(self, embeddings: Tensor) -> Tensor:
        # Conv1D expects `(batch, channels, length)`; the one-hot embedding is `(batch, length, channels)`.
        hidden_state = self.conv(embeddings.transpose(1, 2))
        hidden_state = self.act(hidden_state)
        hidden_state = self.pool(hidden_state)
        hidden_state = self.conv_dropout(hidden_state)
        # Back to `(batch, time, features)` for the TimeDistributed dense and bidirectional LSTM blocks.
        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.time_distributed(hidden_state)
        hidden_state = self.time_distributed_act(hidden_state)
        hidden_state = self.lstm(hidden_state)
        hidden_state = self.recurrent_dropout(hidden_state)
        hidden_state = hidden_state.flatten(start_dim=1)
        hidden_state = self.fc(hidden_state)
        hidden_state = self.fc_act(hidden_state)
        return hidden_state


class DeepMelBidirectionalLstm(nn.Module):
    """Bidirectional LSTM matching the upstream Keras LSTM with `recurrent_activation="hard_sigmoid"`.

    Keras 2.1.5 stores the LSTM kernel as `(input_size, 4 * hidden)` and the recurrent kernel as
    `(hidden, 4 * hidden)`, with gate order `[i, f, c, o]` and a single bias of size `4 * hidden`. We rewrite this
    as a single `(4 * hidden, input_size)` `weight_ih`, `(4 * hidden, hidden)` `weight_hh`, and `(4 * hidden,)`
    `bias` per direction so the converter can map upstream tensors with simple transposes.

    Outputs are concatenated along the feature dimension (matching the upstream
    `Bidirectional(..., merge_mode="concat")`).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_lstm = DeepMelLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.backward_lstm = DeepMelLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        forward = self.forward_lstm(inputs)
        # Keras `Bidirectional` runs the backward LSTM on the time-reversed input, then flips the output back so the
        # forward and backward features for time-step `t` correspond to the same input position before concatenation.
        backward = self.backward_lstm(inputs.flip(1)).flip(1)
        return torch.cat([forward, backward], dim=-1)


class DeepMelLstm(nn.Module):
    """Single-direction LSTM with Keras-style hard-sigmoid recurrent activation."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        # Gate order `[i, f, c, o]` — the same one Keras 2.x stores in its kernel; the converter only needs to
        # transpose the upstream Keras matrices into PyTorch row-major layout.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, time_steps, _ = inputs.shape
        device, dtype = inputs.device, inputs.dtype
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        cell = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.training and self.dropout > 0:
            inputs = inputs * _variational_dropout_mask(
                inputs,
                batch_size=batch_size,
                features=self.input_size,
                dropout=self.dropout,
                broadcast_time=True,
            )
        recurrent_mask = None
        if self.training and self.recurrent_dropout > 0:
            recurrent_mask = _variational_dropout_mask(
                hidden,
                batch_size=batch_size,
                features=self.hidden_size,
                dropout=self.recurrent_dropout,
            )
        # Pre-compute the input projection across all time steps; the recurrent term is the only per-step work left.
        input_proj = F.linear(inputs, self.weight_ih, self.bias)
        outputs = []
        for t in range(time_steps):
            recurrent_hidden = hidden if recurrent_mask is None else hidden * recurrent_mask
            gates = input_proj[:, t] + F.linear(recurrent_hidden, self.weight_hh)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=-1)
            i_gate = _keras_hard_sigmoid(i_gate)
            f_gate = _keras_hard_sigmoid(f_gate)
            o_gate = _keras_hard_sigmoid(o_gate)
            cell = f_gate * cell + i_gate * torch.tanh(c_gate)
            hidden = o_gate * torch.tanh(cell)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


def _variational_dropout_mask(
    inputs: Tensor,
    batch_size: int,
    features: int,
    dropout: float,
    broadcast_time: bool = False,
) -> Tensor:
    shape = (batch_size, 1, features) if broadcast_time else (batch_size, features)
    keep_prob = 1.0 - dropout
    return inputs.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)


def _keras_hard_sigmoid(x: Tensor) -> Tensor:
    """Keras `hard_sigmoid` (`max(0, min(1, 0.2 * x + 0.5))`)."""
    return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)


class DeepMelPooler(nn.Module):
    """Averages the forward and reverse-complement branch features.

    Returns the 256-dim mean of the two per-branch FC representations as the sequence-level pooled embedding.
    `DeepMelForSequencePrediction` reuses the per-branch features (also exposed in the model output) to run the
    final 24-way decoder + sigmoid + averaging that matches the upstream Keras topology exactly.
    """

    def forward(self, forward_features: Tensor, reverse_features: Tensor) -> Tensor:
        return (forward_features + reverse_features) / 2.0


@dataclass
class DeepMelModelOutput(ModelOutput):
    """
    Base class for outputs of the DeepMEL model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, fc_dim)`):
            Per-branch fully-connected representation of the forward DNA strand (i.e. before averaging with the
            reverse-complement branch). Useful for strand-specific interpretation.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, fc_dim)`):
            Branch-averaged sequence-level representation for backbone use cases. The topic head consumes the
            forward and reverse-complement branch representations directly.
        forward_hidden_state (`torch.FloatTensor` of shape `(batch_size, fc_dim)`):
            Fully-connected representation of the forward DNA strand, before branch averaging.
        reverse_hidden_state (`torch.FloatTensor` of shape `(batch_size, fc_dim)`):
            Fully-connected representation of the reverse-complement DNA strand, before branch averaging.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of encoder outputs captured from both the forward and reverse-complement branches.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Always `None`; DeepMEL is a convolutional + recurrent model without explicit attention layers.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    forward_hidden_state: torch.FloatTensor | None = None
    reverse_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# Wire output recording: when output_hidden_states=True the encoder's per-call output is captured.
# DeepMelEncoder is called twice per forward (forward + reverse-complement branch), so hidden_states
# will contain both branch encoder outputs when recording is enabled.
DeepMelPreTrainedModel._can_record_outputs = {"hidden_states": DeepMelEncoder}
