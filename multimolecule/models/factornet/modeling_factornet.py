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
from .configuration_factornet import FactorNetConfig


class FactorNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FactorNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["FactorNetEncoder"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the `_is_hf_initialized`
        # flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class FactorNetModel(FactorNetPreTrainedModel):
    """
    The bare FactorNet backbone. Consumes a fixed-length window of one-hot DNA plus optional auxiliary per-position
    signals (DNase-seq, mappability, ...) and optional per-window metadata features (RNA-seq principal components,
    ...), and returns a pooled per-window representation.

    FactorNet is Siamese: the same convolution / BLSTM / dense stack is applied to the forward strand and to its
    reverse complement, and the two pooled representations are concatenated with the metadata features and averaged
    by the head. `last_hidden_state` is the per-strand-averaged dense representation; `pooler_output` is the same
    tensor and is what the `SequencePredictionHead` consumes.

    Examples:
        >>> import torch
        >>> from multimolecule import FactorNetConfig, FactorNetModel
        >>> config = FactorNetConfig(
        ...     sequence_length=64, conv_kernel_size=8, conv_channels=8,
        ...     pool_size=4, lstm_hidden_size=4, fc_hidden_size=8,
        ...     num_auxiliary_signals=1, num_metadata_features=2,
        ... )
        >>> model = FactorNetModel(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 64))
        >>> aux = torch.randn(1, 64, 1)
        >>> meta = torch.randn(1, 2)
        >>> output = model(input_ids, auxiliary_signal=aux, metadata_features=meta)
        >>> output["pooler_output"].shape
        torch.Size([1, 8])
    """

    def __init__(self, config: FactorNetConfig):
        super().__init__(config)
        self.embeddings = FactorNetEmbeddings(config)
        self.encoder = FactorNetEncoder(config)
        self.head = FactorNetMetadataHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        auxiliary_signal: Tensor | None = None,
        metadata_features: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FactorNetModelOutput:
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
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)  # type: ignore[union-attr]
        self._validate_metadata_features(metadata_features, batch_size)

        forward_embedding, reverse_embedding = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            auxiliary_signal=auxiliary_signal,
        )

        forward_dense = self.encoder(forward_embedding)
        reverse_dense = self.encoder(reverse_embedding)

        forward_fused, reverse_fused = self.head(forward_dense, reverse_dense, metadata_features=metadata_features)
        pooler_output = (forward_fused + reverse_fused) / 2

        return FactorNetModelOutput(
            last_hidden_state=pooler_output,
            pooler_output=pooler_output,
            forward_pooler_output=forward_fused,
            reverse_pooler_output=reverse_fused,
        )

    def _validate_metadata_features(self, metadata_features: Tensor | None, batch_size: int) -> None:
        if self.config.num_metadata_features == 0:
            if metadata_features is not None:
                raise ValueError(
                    "This FactorNet model is configured with num_metadata_features=0 and does not accept a "
                    "`metadata_features` tensor."
                )
            return
        if metadata_features is None:
            raise ValueError(
                f"This FactorNet model is configured with num_metadata_features={self.config.num_metadata_features}; "
                "you must pass the auxiliary `metadata_features` tensor."
            )
        if metadata_features.ndim != 2:
            raise ValueError(
                "`metadata_features` must be a 2D tensor of shape "
                f"(batch_size, {self.config.num_metadata_features}), got shape {tuple(metadata_features.shape)}."
            )
        if metadata_features.size(0) != batch_size:
            raise ValueError(
                f"`metadata_features` batch size ({metadata_features.size(0)}) must match input batch size "
                f"({batch_size})."
            )
        if metadata_features.size(1) != self.config.num_metadata_features:
            raise ValueError(
                f"`metadata_features` last dimension ({metadata_features.size(1)}) must equal "
                f"`config.num_metadata_features` ({self.config.num_metadata_features})."
            )


class FactorNetForSequencePrediction(FactorNetPreTrainedModel):
    """
    FactorNet with the shared `SequencePredictionHead`, predicting per-TF binding probabilities for a fixed-length
    DNA window.

    The head consumes the pooled representation produced by [`FactorNetModel`] and projects it to
    `config.num_labels` logits. The upstream FactorNet release applies a sigmoid activation to these logits; users
    can recover the upstream probabilities via `model.postprocess(...)`.

    Examples:
        >>> import torch
        >>> from multimolecule import FactorNetConfig, FactorNetForSequencePrediction
        >>> config = FactorNetConfig(
        ...     sequence_length=64, conv_kernel_size=8, conv_channels=8,
        ...     pool_size=4, lstm_hidden_size=4, fc_hidden_size=8,
        ...     num_auxiliary_signals=1, num_metadata_features=2, num_labels=1,
        ... )
        >>> model = FactorNetForSequencePrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 64))
        >>> aux = torch.randn(1, 64, 1)
        >>> meta = torch.randn(1, 2)
        >>> output = model(input_ids, auxiliary_signal=aux, metadata_features=meta, labels=torch.zeros(1, 1))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: FactorNetConfig):
        super().__init__(config)
        self.model = FactorNetModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        id2label = getattr(self.config, "id2label", None)
        if id2label is not None:
            labels = [str(id2label.get(index, f"tf_{index}")) for index in range(self.config.num_labels)]
            if any(label != f"LABEL_{index}" for index, label in enumerate(labels)):
                return labels
        if self.config.num_labels == 1:
            return ["binding"]
        return [f"tf_{index}" for index in range(self.config.num_labels)]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        auxiliary_signal: Tensor | None = None,
        metadata_features: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            auxiliary_signal=auxiliary_signal,
            metadata_features=metadata_features,
            return_dict=True,
            **kwargs,
        )

        # Upstream FactorNet averages the *post-sigmoid* probabilities of the forward / reverse strands, not the
        # pre-sigmoid logits. To preserve that semantic, run the shared `SequencePredictionHead.decoder` on each
        # strand's metadata-fused dense projection, take the per-strand sigmoid, average the probabilities, and
        # convert back to a logit so the rest of the MultiMolecule contract (loss expects logits) is unchanged.
        forward_outputs = {"pooler_output": outputs.forward_pooler_output}
        reverse_outputs = {"pooler_output": outputs.reverse_pooler_output}
        forward_logits = self.sequence_head(forward_outputs, labels=None).logits
        reverse_logits = self.sequence_head(reverse_outputs, labels=None).logits
        average_probability = (torch.sigmoid(forward_logits) + torch.sigmoid(reverse_logits)) / 2
        # Inverse sigmoid; clamp to avoid `inf` for probabilities pinned to 0 / 1.
        eps = torch.finfo(average_probability.dtype).eps
        clamped = average_probability.clamp(min=eps, max=1.0 - eps)
        logits = torch.log(clamped) - torch.log1p(-clamped)

        loss = None
        if labels is not None:
            loss = self.sequence_head.criterion(logits, labels)

        return SequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def postprocess(self, outputs: Any) -> Tensor:
        return torch.sigmoid(outputs["logits"])


class FactorNetEmbeddings(nn.Module):
    """One-hot embedding for FactorNet.

    Produces a `(forward, reverse_complement)` pair of `(batch_size, num_input_channels, sequence_length)` tensors.
    The DNA one-hot is derived from `input_ids` using the MultiMolecule DNA token order; the reverse complement is
    formed by reversing the sequence axis (and swapping the canonical A<->T / C<->G channels). Auxiliary per-position
    signals (e.g. DNase, mappability) are concatenated on the channel axis and reversed only along the sequence axis
    for the reverse-strand branch.
    """

    def __init__(self, config: FactorNetConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
        self.num_auxiliary_signals = config.num_auxiliary_signals
        # The MultiMolecule DNA token order is ["A", "C", "G", "T"]; reverse-complement therefore swaps
        # 0 <-> 3 (A <-> T) and 1 <-> 2 (C <-> G). For non-standard vocab sizes the converter is responsible for
        # mapping input channels; for the default ACGT alphabet we apply this permutation in-place.
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16) so F.one_hot output
        # (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        auxiliary_signal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            self._check_sequence_length(input_ids.size(-1))
            one_hot = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                one_hot = one_hot * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_sequence_length(inputs_embeds.size(1))
            one_hot = inputs_embeds.to(dtype)
        if attention_mask is not None:
            one_hot = one_hot * attention_mask.unsqueeze(-1).to(one_hot.dtype)
        # `one_hot` is `(batch, sequence_length, vocab_size)`; transpose so concatenation happens on the channel axis.
        forward_dna = one_hot.transpose(1, 2)
        reverse_dna = self._reverse_complement_dna(forward_dna)
        if self.num_auxiliary_signals > 0:
            self._check_auxiliary_signal(auxiliary_signal, forward_dna.size(0))
            assert auxiliary_signal is not None  # narrowed by `_check_auxiliary_signal`
            forward_aux = auxiliary_signal.to(device=forward_dna.device, dtype=forward_dna.dtype)
            forward_aux = forward_aux.transpose(1, 2)
            reverse_aux = torch.flip(forward_aux, dims=(-1,))
            forward_embedding = torch.cat([forward_dna, forward_aux], dim=1)
            reverse_embedding = torch.cat([reverse_dna, reverse_aux], dim=1)
        else:
            if auxiliary_signal is not None:
                raise ValueError(
                    "This FactorNet model is configured with num_auxiliary_signals=0 and does not accept an "
                    "`auxiliary_signal` tensor."
                )
            forward_embedding = forward_dna
            reverse_embedding = reverse_dna
        return forward_embedding, reverse_embedding

    def _reverse_complement_dna(self, forward_dna: Tensor) -> Tensor:
        if forward_dna.size(1) != 4:
            # For non-ACGT vocabularies, the converter is responsible for setting up appropriate weights; we still
            # flip the sequence axis but skip the channel swap, since the semantic complement is undefined.
            return torch.flip(forward_dna, dims=(-1,))
        # ACGT -> TGCA (i.e. reverse channels) then reverse along the length axis: this is the reverse complement.
        return torch.flip(forward_dna, dims=(1, -1))

    def _check_sequence_length(self, sequence_length: int):
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"FactorNet expects fixed-length {self.sequence_length} bp inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )

    def _check_auxiliary_signal(self, auxiliary_signal: Tensor | None, batch_size: int) -> None:
        if auxiliary_signal is None:
            raise ValueError(
                f"This FactorNet model is configured with num_auxiliary_signals={self.num_auxiliary_signals}; "
                "you must pass the per-position `auxiliary_signal` tensor of shape "
                f"(batch_size, {self.sequence_length}, {self.num_auxiliary_signals})."
            )
        if auxiliary_signal.ndim != 3:
            raise ValueError(
                "`auxiliary_signal` must be a 3D tensor of shape "
                f"(batch_size, {self.sequence_length}, {self.num_auxiliary_signals}), got shape "
                f"{tuple(auxiliary_signal.shape)}."
            )
        if auxiliary_signal.size(0) != batch_size:
            raise ValueError(
                f"`auxiliary_signal` batch size ({auxiliary_signal.size(0)}) must match input batch size "
                f"({batch_size})."
            )
        if auxiliary_signal.size(1) != self.sequence_length:
            raise ValueError(
                f"`auxiliary_signal` sequence length ({auxiliary_signal.size(1)}) must equal "
                f"`config.sequence_length` ({self.sequence_length})."
            )
        if auxiliary_signal.size(2) != self.num_auxiliary_signals:
            raise ValueError(
                f"`auxiliary_signal` last dimension ({auxiliary_signal.size(2)}) must equal "
                f"`config.num_auxiliary_signals` ({self.num_auxiliary_signals})."
            )


class FactorNetEncoder(nn.Module):
    """Per-strand convolution + BLSTM + dense stack.

    Reproduces the upstream `Conv1D -> Dropout -> TimeDistributed(Dense) -> MaxPool1D -> Bidirectional(LSTM) ->
    Dropout -> Flatten -> Dense` pipeline. The same encoder instance is applied to the forward and the
    reverse-complement embeddings.
    """

    def __init__(self, config: FactorNetConfig):
        super().__init__()
        in_channels = config.num_input_channels
        self.conv1 = nn.Conv1d(in_channels, config.conv_channels, kernel_size=config.conv_kernel_size)
        self.act1 = ACT2FN[config.hidden_act]
        self.dropout1 = nn.Dropout(config.conv_dropout)
        # `TimeDistributed(Dense)` is a pointwise (kernel=1) linear projection along the sequence axis.
        self.pointwise = nn.Linear(config.conv_channels, config.conv_channels)
        self.act_pointwise = ACT2FN[config.hidden_act]
        self.pool = nn.MaxPool1d(kernel_size=config.pool_size, stride=config.pool_size)
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm: FactorNetBidirectionalHardSigmoidLstm | None
        if config.lstm_hidden_size > 0:
            # FactorNet trains with Keras 1.1's `LSTM(inner_activation="hard_sigmoid", activation="tanh")`. PyTorch's
            # `nn.LSTM` uses regular `sigmoid` for the gates, so a checkpoint-faithful conversion needs a custom LSTM
            # that applies `hard_sigmoid` to the input/forget/output gates and `tanh` to the cell and output state.
            self.lstm = FactorNetBidirectionalHardSigmoidLstm(
                input_size=config.conv_channels,
                hidden_size=config.lstm_hidden_size,
            )
        else:
            self.lstm = None
        self.dropout2 = nn.Dropout(config.post_lstm_dropout)
        self.dense = nn.Linear(config.flattened_size, config.fc_hidden_size)
        self.act2 = ACT2FN[config.hidden_act]

    def forward(self, embedding: Tensor) -> Tensor:
        # `embedding` is (batch, num_input_channels, sequence_length).
        hidden_state = self.conv1(embedding)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        # Pointwise dense applied along the channel axis at every timestep.
        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.pointwise(hidden_state)
        hidden_state = self.act_pointwise(hidden_state)
        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.pool(hidden_state)
        # The custom hard-sigmoid BiLSTM and the identity fall-through both expect (batch, seq, features).
        hidden_state = hidden_state.transpose(1, 2)
        if self.lstm is not None:
            hidden_state = self.lstm(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        hidden_state = hidden_state.flatten(1)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.act2(hidden_state)
        return hidden_state


class FactorNetBidirectionalHardSigmoidLstm(nn.Module):
    """Bidirectional LSTM with Keras 1.1 gating semantics (hard sigmoid for gates, tanh for cell / output).

    Keras 1.1 `LSTM(activation="tanh", inner_activation="hard_sigmoid")` differs from `torch.nn.LSTM` only in the
    gate activation: it applies `hard_sigmoid(x) = clip(0.2 * x + 0.5, 0, 1)` instead of the standard `sigmoid`.
    This module reproduces that behavior so the FactorNet checkpoint can be loaded with bit-for-bit (gating)
    semantics. State dict keys follow `torch.nn.LSTM`'s naming: `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`,
    `bias_hh_l0`, plus a `_reverse` suffix for the backward direction. Gate order is `(input, forget, cell, output)`,
    matching the PyTorch convention; the converter is responsible for fusing the per-gate Keras tensors in this
    order.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Names match `nn.LSTM` so the converter can fill them with simple `weight_ih_l0` / `weight_hh_l0` keys.
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size))
        self.weight_ih_l0_reverse = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0_reverse = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih_l0_reverse = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh_l0_reverse = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)

    def forward(self, hidden_state: Tensor) -> Tensor:
        forward = self._direction(
            hidden_state,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            reverse=False,
        )
        backward = self._direction(
            hidden_state,
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
            reverse=True,
        )
        return torch.cat([forward, backward], dim=-1)

    def _direction(
        self,
        hidden_state: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
        reverse: bool,
    ) -> Tensor:
        # `hidden_state` is `(batch, seq, input_size)`.
        batch_size, seq_len, _ = hidden_state.shape
        if reverse:
            hidden_state = torch.flip(hidden_state, dims=(1,))
        # Compute all input projections in one matmul along the time axis.
        ih_proj = F.linear(hidden_state, weight_ih, bias_ih + bias_hh)
        h = hidden_state.new_zeros(batch_size, self.hidden_size)
        c = hidden_state.new_zeros(batch_size, self.hidden_size)
        outputs: list[Tensor] = []
        for t in range(seq_len):
            gates = ih_proj[:, t, :] + F.linear(h, weight_hh)
            i, f, g, o = gates.chunk(4, dim=-1)
            # Keras `hard_sigmoid` = clip(0.2 * x + 0.5, 0, 1).
            i = torch.clamp(0.2 * i + 0.5, min=0.0, max=1.0)
            f = torch.clamp(0.2 * f + 0.5, min=0.0, max=1.0)
            o = torch.clamp(0.2 * o + 0.5, min=0.0, max=1.0)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
        out = torch.stack(outputs, dim=1)
        if reverse:
            out = torch.flip(out, dims=(1,))
        return out


class FactorNetMetadataHead(nn.Module):
    """Per-strand metadata fusion.

    Reproduces the upstream second-dense projection: for each strand, the encoder's `(fc_hidden_size,)` pooled
    representation is concatenated with the per-window metadata features and projected back to `fc_hidden_size`.
    The forward / reverse strand outputs are returned separately so the downstream sigmoid head can average the
    per-strand binding probabilities (upstream `merge` mode `ave`).
    """

    def __init__(self, config: FactorNetConfig):
        super().__init__()
        in_features = config.fc_hidden_size + config.num_metadata_features
        self.dense = nn.Linear(in_features, config.fc_hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.num_metadata_features = config.num_metadata_features

    def forward(
        self,
        forward_dense: Tensor,
        reverse_dense: Tensor,
        metadata_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.num_metadata_features > 0:
            metadata = metadata_features.to(  # type: ignore[union-attr]
                device=forward_dense.device, dtype=forward_dense.dtype
            )
            forward_input = torch.cat([forward_dense, metadata], dim=-1)
            reverse_input = torch.cat([reverse_dense, metadata], dim=-1)
        else:
            forward_input = forward_dense
            reverse_input = reverse_dense
        forward_output = self.act(self.dense(forward_input))
        reverse_output = self.act(self.dense(reverse_input))
        return forward_output, reverse_output


@dataclass
class FactorNetModelOutput(ModelOutput):
    """
    Base class for outputs of the FactorNet backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, fc_hidden_size)`):
            Pooled per-window representation produced by averaging the forward and reverse-complement metadata-fused
            dense projections. This is the same tensor exposed as `pooler_output`.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, fc_hidden_size)`):
            Strand-averaged metadata-fused dense projection; the tensor exposed for downstream features that don't
            need per-strand probabilities.
        forward_pooler_output (`torch.FloatTensor` of shape `(batch_size, fc_hidden_size)`):
            Forward-strand metadata-fused dense projection (i.e. the metadata-fused dense output of the forward
            encoder pass, before the per-TF sigmoid head). Consumed by
            [`FactorNetForSequencePrediction`][multimolecule.models.FactorNetForSequencePrediction] to compute the
            upstream-equivalent per-strand-sigmoid-then-average binding probability.
        reverse_pooler_output (`torch.FloatTensor` of shape `(batch_size, fc_hidden_size)`):
            Reverse-complement strand metadata-fused dense projection. Pair to `forward_pooler_output`.
        hidden_states (always `None`):
            FactorNet collapses the sequence dimension through its dense layers; no per-layer hidden states are
            recorded. Present only for compatibility with the Transformers output convention.
        attentions (always `None`):
            FactorNet has no attention layers; this field is always `None` and is present only for compatibility
            with the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    forward_pooler_output: torch.FloatTensor | None = None
    reverse_pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
