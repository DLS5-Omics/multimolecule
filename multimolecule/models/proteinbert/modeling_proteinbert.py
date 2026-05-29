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
from typing import Any

import torch
import torch.nn.functional as F
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import init
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults

from multimolecule.modules import HeadConfig, MaskedLMHead, SequencePredictionHead, TokenPredictionHead

from ..modeling_outputs import SequencePredictorOutput, TokenPredictorOutput
from .configuration_proteinbert import ProteinBertConfig


class ProteinBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProteinBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ProteinBertLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, ProteinBertGlobalAttention):
            init.xavier_uniform_(module.query)
            init.xavier_uniform_(module.key)
            init.xavier_uniform_(module.value)


class ProteinBertModel(ProteinBertPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProteinBertConfig, ProteinBertModel, ProteinTokenizer
        >>> config = ProteinBertConfig()
        >>> model = ProteinBertModel(config)
        >>> tokenizer = ProteinTokenizer.from_pretrained("multimolecule/protein")
        >>> input = tokenizer("MVLSPADKT", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 11, 128])
        >>> output["pooler_output"].shape
        torch.Size([1, 512])
    """

    def __init__(self, config: ProteinBertConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = ProteinBertEmbeddings(config)
        self.encoder = ProteinBertEncoder(config)

        # Initialize weights and apply final processing
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
        annotations: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | ProteinBertModelOutput:
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

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        if attention_mask is None:
            if input_ids is not None and self.pad_token_id is not None:
                attention_mask = input_ids.ne(self.pad_token_id)
            else:
                attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if annotations is None:
            annotations = hidden_states.new_zeros(hidden_states.shape[0], self.config.annotation_size)
        annotations = annotations.to(device=hidden_states.device, dtype=hidden_states.dtype)
        global_states = self.embeddings.project_annotations(annotations)

        encoder_outputs = self.encoder(
            hidden_states,
            global_states,
            attention_mask=attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", self.config.output_hidden_states),
            output_attentions=kwargs.get("output_attentions", self.config.output_attentions),
        )

        return ProteinBertModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ProteinBertForSequencePrediction(ProteinBertPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProteinBertConfig, ProteinBertForSequencePrediction
        >>> config = ProteinBertConfig()
        >>> model = ProteinBertForSequencePrediction(config)
        >>> input_ids = torch.tensor([[1, 16, 23, 15, 21, 18, 6, 9, 14, 2]])
        >>> output = model(input_ids, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: ProteinBertConfig):
        super().__init__(config)
        self.model = ProteinBertModel(config)
        head_config = HeadConfig(config.head or {})
        if head_config.hidden_size is None:
            # ProteinBert exposes two feature streams of different width: the per-token `last_hidden_state`
            # (hidden_size) and the global `pooler_output` (global_hidden_size). Sequence heads read the pooled
            # stream by default, so any output other than `last_hidden_state` resolves to global_hidden_size.
            head_config.hidden_size = (
                config.hidden_size if head_config.output_name == "last_hidden_state" else config.global_hidden_size
            )
        self.sequence_head = SequencePredictionHead(config, head_config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        annotations: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            annotations=annotations,
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


class ProteinBertForTokenPrediction(ProteinBertPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProteinBertConfig, ProteinBertForTokenPrediction
        >>> config = ProteinBertConfig()
        >>> model = ProteinBertForTokenPrediction(config)
        >>> input_ids = torch.tensor([[1, 16, 23, 15, 21, 18, 6, 9, 14, 2]])
        >>> output = model(input_ids, labels=torch.randint(2, (1, 8)))
        >>> output["logits"].shape
        torch.Size([1, 8, 1])
    """

    def __init__(self, config: ProteinBertConfig):
        super().__init__(config)
        self.model = ProteinBertModel(config)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        annotations: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            annotations=annotations,
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


class ProteinBertForMaskedLM(ProteinBertPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProteinBertConfig, ProteinBertForMaskedLM
        >>> config = ProteinBertConfig()
        >>> model = ProteinBertForMaskedLM(config)
        >>> input_ids = torch.tensor([[1, 16, 23, 15, 21, 18, 6, 9, 14, 2]])
        >>> output = model(input_ids, labels=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 10, 37])
    """

    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        tied_weights = super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        if all_submodels:
            return tied_weights
        return tied_weights | self._tied_weights_keys

    def __init__(self, config: ProteinBertConfig):
        super().__init__(config)
        self.model = ProteinBertModel(config)
        self.lm_head = MaskedLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings
        if hasattr(self.lm_head, "bias"):
            self.lm_head.bias = embeddings.bias

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        annotations: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            annotations=annotations,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.lm_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ProteinBertForPreTraining(ProteinBertPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ProteinBertConfig, ProteinBertForPreTraining
        >>> config = ProteinBertConfig()
        >>> model = ProteinBertForPreTraining(config)
        >>> input_ids = torch.tensor([[1, 16, 23, 15, 21, 18, 6, 9, 14, 2]])
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([1, 10, 37])
        >>> output["annotation_logits"].shape
        torch.Size([1, 8943])
    """

    _tied_weights_keys = {
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        tied_weights = super().get_expanded_tied_weights_keys(all_submodels=all_submodels)
        if all_submodels:
            return tied_weights
        return tied_weights | self._tied_weights_keys

    def __init__(self, config: ProteinBertConfig):
        super().__init__(config)
        self.model = ProteinBertModel(config)
        self.lm_head = MaskedLMHead(config)
        self.annotation_head = ProteinBertAnnotationPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings
        if hasattr(self.lm_head, "bias"):
            self.lm_head.bias = embeddings.bias

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        annotations: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        annotation_labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | ProteinBertForPreTrainingOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            annotations=annotations,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        lm_output = self.lm_head(outputs, labels)
        annotation_logits = self.annotation_head(outputs.pooler_output)

        loss = lm_output.loss
        if annotation_labels is not None:
            annotation_loss = F.binary_cross_entropy_with_logits(annotation_logits, annotation_labels.float())
            loss = annotation_loss if loss is None else loss + annotation_loss

        return ProteinBertForPreTrainingOutput(
            loss=loss,
            logits=lm_output.logits,
            annotation_logits=annotation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ProteinBertEmbeddings(nn.Module):
    def __init__(self, config: ProteinBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.annotation_embeddings = nn.Linear(config.annotation_size, config.global_hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return inputs_embeds

    def project_annotations(self, annotations: Tensor) -> Tensor:
        return self.activation(self.annotation_embeddings(annotations))


class ProteinBertEncoder(nn.Module):
    def __init__(self, config: ProteinBertConfig):
        super().__init__()
        self.layer = nn.ModuleList([ProteinBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        global_states: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ProteinBertModelOutput:
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        all_attentions: list[Tensor] | None = [] if output_attentions else None
        input_mask = (
            attention_mask.unsqueeze(-1).to(device=hidden_states.device, dtype=hidden_states.dtype)
            if attention_mask is not None
            else None
        )

        for layer_module in self.layer:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, global_states, attention_probs = layer_module(
                hidden_states,
                global_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if input_mask is not None:
                hidden_states = hidden_states * input_mask
            if all_attentions is not None and attention_probs is not None:
                all_attentions.append(attention_probs)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return ProteinBertModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=global_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )


class ProteinBertLayer(GradientCheckpointingLayer):
    def __init__(self, config: ProteinBertConfig):
        super().__init__()
        self.global_to_sequence = nn.Linear(config.global_hidden_size, config.hidden_size)
        self.narrow_conv = ProteinBertConvBranch(config, dilation=1)
        self.wide_conv = ProteinBertConvBranch(config, dilation=config.wide_conv_dilation_rate)
        self.sequence_layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.sequence_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.sequence_layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.global_dense1 = nn.Linear(config.global_hidden_size, config.global_hidden_size)
        self.global_attention = ProteinBertGlobalAttention(config)
        self.global_layer_norm1 = nn.LayerNorm(config.global_hidden_size, eps=config.layer_norm_eps)
        self.global_dense2 = nn.Linear(config.global_hidden_size, config.global_hidden_size)
        self.global_layer_norm2 = nn.LayerNorm(config.global_hidden_size, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Tensor,
        global_states: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        sequence_from_global = self.activation(self.global_to_sequence(global_states)).unsqueeze(1)
        narrow_conv = self.narrow_conv(hidden_states)
        wide_conv = self.wide_conv(hidden_states)
        hidden_states = self.sequence_layer_norm1(hidden_states + sequence_from_global + narrow_conv + wide_conv)

        sequence_dense = self.activation(self.sequence_dense(hidden_states))
        hidden_states = self.sequence_layer_norm2(hidden_states + sequence_dense)

        global_dense = self.activation(self.global_dense1(global_states))
        attention_output, attention_probs = self.global_attention(
            global_states,
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        global_states = self.global_layer_norm1(global_states + global_dense + attention_output)

        global_dense = self.activation(self.global_dense2(global_states))
        global_states = self.global_layer_norm2(global_states + global_dense)

        return hidden_states, global_states, attention_probs


class ProteinBertConvBranch(nn.Module):
    def __init__(self, config: ProteinBertConfig, dilation: int):
        super().__init__()
        total_padding = dilation * (config.conv_kernel_size - 1)
        padding_left = total_padding // 2
        self.padding = (padding_left, total_padding - padding_left)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            dilation=dilation,
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        if self.padding != (0, 0):
            hidden_states = F.pad(hidden_states, self.padding)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return self.activation(hidden_states)


class ProteinBertGlobalAttention(nn.Module):
    def __init__(self, config: ProteinBertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_key_size = config.attention_key_size
        self.attention_value_size = config.global_hidden_size // config.num_attention_heads
        self.query = nn.Parameter(
            torch.empty(config.num_attention_heads, config.global_hidden_size, config.attention_key_size)
        )
        self.key = nn.Parameter(torch.empty(config.num_attention_heads, config.hidden_size, config.attention_key_size))
        self.value = nn.Parameter(
            torch.empty(config.num_attention_heads, config.hidden_size, self.attention_value_size)
        )
        self.activation = ACT2FN[config.hidden_act]
        self.scaling = config.attention_key_size**-0.5

    def forward(
        self,
        global_states: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        query_states = torch.tanh(torch.einsum("bg,hgk->bhk", global_states, self.query))
        key_states = torch.tanh(torch.einsum("bls,hsk->bhlk", hidden_states, self.key))
        value_states = self.activation(torch.einsum("bls,hsv->bhlv", hidden_states, self.value))

        attention_scores = torch.einsum("bhk,bhlk->bhl", query_states, key_states) * self.scaling
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask[:, None, :].to(device=attention_scores.device, dtype=torch.bool),
                torch.finfo(attention_scores.dtype).min,
            )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context = torch.einsum("bhl,bhlv->bhv", attention_probs, value_states)
        context = context.reshape(global_states.shape[0], self.num_attention_heads * self.attention_value_size)

        return context, attention_probs if output_attentions else None


class ProteinBertAnnotationPredictionHead(nn.Module):
    def __init__(self, config: ProteinBertConfig):
        super().__init__()
        self.decoder = nn.Linear(config.global_hidden_size, config.annotation_size)

    def forward(self, global_states: Tensor) -> Tensor:
        return self.decoder(global_states)


@dataclass
class ProteinBertModelOutput(ModelOutput):
    """
    Base class for ProteinBERT backbone outputs.

    Args:
        last_hidden_state:
            Local residue representations of shape `(batch_size, sequence_length, hidden_size)`.
        pooler_output:
            Global protein representations of shape `(batch_size, global_hidden_size)`.
        hidden_states:
            Hidden states of the local representation stack.
        attentions:
            Global-attention probabilities for each layer.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ProteinBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ProteinBertForPreTraining`].

    Args:
        loss:
            Masked language modeling plus annotation prediction loss.
        logits:
            Prediction scores of the language modeling head.
        annotation_logits:
            Prediction scores of the Gene Ontology annotation head.
        hidden_states:
            Hidden states of the local representation stack.
        attentions:
            Global-attention probabilities for each layer.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    annotation_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
