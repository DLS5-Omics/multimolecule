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

import torch
import torch.nn.functional as F
from danling import NestedTensor
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults

from multimolecule.modules import ContactPredictionHead, HeadOutput, SequencePredictionHead, TokenPredictionHead

from ..modeling_outputs import ContactPredictorOutput, SequencePredictorOutput, TokenPredictorOutput
from .configuration_carp import CarpConfig


class CarpPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CarpConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CarpLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # CARP's reference implementation uses PyTorch module defaults. Keep
        # those constructor initializers instead of applying a Transformers
        # normal initialization pass.
        return


class CarpModel(CarpPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import ProteinTokenizer
        >>> from multimolecule.models.carp import CarpConfig, CarpModel
        >>> config = CarpConfig()
        >>> model = CarpModel(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> inputs = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**inputs)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 11, 128])
        >>> output["pooler_output"].shape
        torch.Size([1, 128])
    """

    def __init__(self, config: CarpConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = CarpEmbeddings(config)
        self.encoder = CarpEncoder(config)
        self.pooler = CarpPooler() if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @can_return_tuple
    @merge_with_config_defaults
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | CarpModelOutput:
        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", self.config.output_hidden_states),
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output, attention_mask) if self.pooler is not None else None

        return CarpModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CarpForSequencePrediction(CarpPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.carp import CarpConfig, CarpForSequencePrediction
        >>> config = CarpConfig()
        >>> model = CarpForSequencePrediction(config)
        >>> inputs = torch.tensor([[1, 6, 23, 15, 21, 18, 6, 8, 14, 22, 2]])
        >>> output = model(inputs, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: CarpConfig):
        super().__init__(config)
        self.model = CarpModel(config)
        self.num_labels = config.num_labels
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        self.post_init()

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


class CarpForTokenPrediction(CarpPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.carp import CarpConfig, CarpForTokenPrediction
        >>> config = CarpConfig()
        >>> model = CarpForTokenPrediction(config)
        >>> inputs = torch.tensor([[1, 6, 23, 15, 21, 18, 6, 8, 14, 22, 2]])
        >>> output = model(inputs, labels=torch.randint(2, (1, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 1])
    """

    def __init__(self, config: CarpConfig):
        super().__init__(config)
        self.model = CarpModel(config, add_pooling_layer=False)
        self.num_labels = config.num_labels
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return TokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CarpForContactPrediction(CarpPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.carp import CarpConfig, CarpForContactPrediction
        >>> config = CarpConfig()
        >>> model = CarpForContactPrediction(config)
        >>> inputs = torch.tensor([[1, 6, 23, 15, 21, 18, 6, 8, 14, 22, 2]])
        >>> output = model(inputs, labels=torch.randint(2, (1, 9, 9)))
        >>> output["logits"].shape
        torch.Size([1, 9, 9, 1])
    """

    def __init__(self, config: CarpConfig):
        super().__init__(config)
        self.model = CarpModel(config, add_pooling_layer=False)
        self.num_labels = config.num_labels
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        if self.contact_head.require_attentions:
            raise ValueError("CARP does not expose attention maps; use a representation-based contact head.")

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | ContactPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return ContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CarpForMaskedLM(CarpPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models.carp import CarpConfig, CarpForMaskedLM
        >>> config = CarpConfig()
        >>> model = CarpForMaskedLM(config)
        >>> inputs = torch.tensor([[1, 6, 23, 15, 21, 18, 6, 8, 14, 22, 2]])
        >>> output = model(inputs, labels=inputs)
        >>> output["logits"].shape
        torch.Size([1, 11, 37])
    """

    def __init__(self, config: CarpConfig):
        super().__init__(config)
        self.model = CarpModel(config, add_pooling_layer=False)
        self.lm_head = CarpLMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.word_embeddings = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.lm_head(outputs.last_hidden_state, labels)
        logits, loss = output.logits, output.loss

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CarpForPreTraining(CarpForMaskedLM):
    pass


class CarpEmbeddings(nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            # CARP's upstream tokenizer uses a zero vector for the mask token.
            padding_idx=config.mask_token_id,
        )
        self.projection = CarpPositionWiseLinear(config.embedding_size, config.hidden_size)

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return self.projection(inputs_embeds)


class CarpEncoder(nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        dilation_cycle = []
        dilation = 1
        while dilation < config.max_dilation:
            dilation_cycle.append(dilation)
            dilation *= 2
        dilation_cycle.append(config.max_dilation)
        dilations = [dilation_cycle[index % len(dilation_cycle)] for index in range(config.num_hidden_layers)]
        self.layer = nn.ModuleList([CarpLayer(config, dilation=dilation) for dilation in dilations])
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> CarpEncoderOutput:
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        input_mask = None
        if attention_mask is not None:
            input_mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype, device=hidden_states.device)

        for layer_module in self.layer:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer_module(hidden_states, input_mask=input_mask)
            hidden_states = self.dropout(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return CarpEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=None,
        )


class CarpLayer(GradientCheckpointingLayer):
    def __init__(self, config: CarpConfig, dilation: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = CarpPositionWiseLinear(config.hidden_size, config.intermediate_size)
        self.layer_norm2 = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)
        self.convolution = CarpMaskedConv1d(
            config.intermediate_size,
            config.intermediate_size,
            kernel_size=config.kernel_size,
            dilation=dilation,
        )
        self.layer_norm3 = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)
        self.output = CarpPositionWiseLinear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor, input_mask: Tensor | None = None) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.convolution(hidden_states, input_mask=input_mask)
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output(hidden_states)
        return residual + hidden_states


class CarpMaskedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, hidden_states: Tensor, input_mask: Tensor | None = None) -> Tensor:
        if input_mask is not None:
            hidden_states = hidden_states * input_mask
        return super().forward(hidden_states.transpose(1, 2)).transpose(1, 2)


class CarpPositionWiseLinear(nn.Conv1d):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states.transpose(1, 2)).transpose(1, 2)


class CarpPooler(nn.Module):
    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(device=hidden_states.device, dtype=hidden_states.dtype)
        pooled_output = (hidden_states * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(torch.finfo(hidden_states.dtype).eps)
        return pooled_output / denominator


class CarpLMHead(nn.Module):
    r"""
    Masked language modeling head for CARP.

    CARP applies layer normalization directly before the output projection with no intermediate transform. The
    shared [`MaskedLMHead`][multimolecule.modules.MaskedLMHead] has no layer-norm-only transform path, so a small
    model-local head is used to preserve the original checkpoint behavior.
    """

    def __init__(self, config: CarpConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states: Tensor, labels: Tensor | None = None) -> HeadOutput:
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return HeadOutput(logits, loss)
        return HeadOutput(logits)


@dataclass
class CarpEncoderOutput(ModelOutput):
    """
    Base class for outputs of the CARP convolutional encoder.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden states at the output of the last encoder layer.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the embedding output plus one after each encoder layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        attentions:
            Always `None`; CARP is a convolutional model and has no attention layers. Provided for compatibility with
            the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class CarpModelOutput(ModelOutput):
    """
    Base class for outputs of the CARP backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden states at the output of the last encoder layer.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Mean-pooled sequence representation over unmasked tokens.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the embedding output plus one after each encoder layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        attentions:
            Always `None`; CARP is a convolutional model and has no attention layers. Provided for compatibility with
            the Transformers output convention.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
