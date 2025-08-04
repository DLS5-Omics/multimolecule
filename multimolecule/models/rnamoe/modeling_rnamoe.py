# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple
from warnings import warn

import torch
from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

from multimolecule.modules import (
    ContactPredictionHead,
    ContactPredictionResNetHead,
    HeadConfig,
    MaskedLMHead,
    SequencePredictionHead,
    SinusoidalEmbedding,
    TokenPredictionHead,
)

from .configuration_rnamoe import RnaMoeConfig

logger = logging.get_logger(__name__)


class RnaMoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RnaMoeConfig
    base_model_prefix = "rnamoe"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RnaMoeLayer", "RnaMoeEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RnaMoeModel(RnaMoePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMoeConfig, RnaMoeModel, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 768])
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    pairwise_bias_map: Tensor

    def __init__(
        self, config: RnaMoeConfig, add_pooling_layer: bool = True, tokenizer: PreTrainedTokenizer | None = None
    ):
        super().__init__(config)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("multimolecule/rna")
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
        if self.vocab_size != config.vocab_size:
            raise ValueError(
                f"Vocab size in tokenizer ({self.vocab_size}) does not match the one in config ({config.vocab_size})"
            )
        token_to_ids = self.tokenizer._token_to_id
        tokens = sorted(token_to_ids, key=token_to_ids.get)
        pairwise_bias_dict = get_pairwise_bias_dict(config.pairwise_alpha)
        self.register_buffer(
            "pairwise_bias_map",
            torch.tensor([[pairwise_bias_dict.get(f"{i}{j}", 0) for i in tokens] for j in tokens]),
            persistent=False,
        )
        self.pairwise_bias_proj = nn.Sequential(
            nn.Linear(1, config.num_attention_heads // 2),
            nn.GELU(),
            nn.Linear(config.num_attention_heads // 2, config.num_attention_heads),
        )
        self.embeddings = RnaMoeEmbeddings(config)
        self.encoder = RnaMoeEncoder(config)
        self.pooler = RnaMoePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_pairwise_bias(
        self, input_ids: Tensor | NestedTensor, attention_mask: Tensor | NestedTensor | None = None
    ) -> Tensor | NestedTensor:
        batch_size, seq_len = input_ids.shape

        # Broadcasting data indices to compute indices
        data_index_x = input_ids.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        data_index_y = input_ids.unsqueeze(1).expand(batch_size, seq_len, seq_len)

        # Get bias from pairwise_bias_map
        return self.pairwise_bias_map[data_index_x, data_index_y]

        # Zhiyuan: Is it really necessary to mask the bias?
        # The mask position should have been nan, and the implementation is incorrect anyway
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        #     bias = bias * attention_mask

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeModelOutputWithPooling:
        r"""
        Args:
            encoder_hidden_states:
                Shape: `(batch_size, sequence_length, hidden_size)`

                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask:
                Shape: `(batch_size, sequence_length)`

                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values:
                Tuple of length `config.n_layers` with each tuple having 4 tensors of shape
                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)

                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        """
        if kwargs:
            warn(
                f"Additional keyword arguments `{', '.join(kwargs)}` are detected in "
                f"`{self.__class__.__name__}.forward`, they will be ignored.\n"
                "This is provided for backward compatibility and may lead to unexpected behavior."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attention_biases = (
            output_attention_biases if output_attention_biases is not None else self.config.output_attention_biases
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pairwise_bias = self.get_pairwise_bias(input_ids, attention_mask)
        attention_bias = self.pairwise_bias_proj(pairwise_bias.unsqueeze(-1)).transpose(1, 3)

        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore[union-attr]

        if attention_mask is None:
            attention_mask = (
                input_ids.ne(self.pad_token_id)
                if self.pad_token_id is not None
                else torch.ones(((batch_size, seq_length)), device=device)
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            attention_bias=attention_bias,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_attention_biases=output_attention_biases,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return RnaMoeModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attention_biases=encoder_outputs.attention_biases,
            attentions=encoder_outputs.attentions,
            attention_queries=encoder_outputs.attention_queries,
            attention_keys=encoder_outputs.attention_keys,
            attention_values=encoder_outputs.attention_values,
            router_probs=encoder_outputs.router_probs,
        )


class RnaMoeForSequencePrediction(RnaMoePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMoeConfig, RnaMoeForSequencePrediction, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        self.rnamoe = RnaMoeModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeSequencePredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_attention_biases=output_attention_biases,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMoeSequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeForTokenPrediction(RnaMoePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMoeConfig, RnaMoeForTokenPrediction, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.rnamoe = RnaMoeModel(config, add_pooling_layer=True)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeTokenPredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_attention_biases=output_attention_biases,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMoeTokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeForContactPrediction(RnaMoePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMoeConfig, RnaMoeForContactPrediction, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        self.rnamoe = RnaMoeModel(config, add_pooling_layer=True)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeContactPredictorOutput:
        if output_attentions is False:
            warn("output_attentions must be True for contact classification and will be ignored.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMoeContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeForNucleotidePrediction(RnaMoeForTokenPrediction):

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        warn(
            "`RnaMoeForNucleotidePrediction` is deprecated and will be removed in 0.0.6. "
            "Please use `CaLmForTokenPrediction` instead.",
            DeprecationWarning,
        )


class RnaMoeForMaskedLM(RnaMoePreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMoeConfig, RnaMoeForMaskedLM, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        self.rnamoe = RnaMoeModel(config, add_pooling_layer=False)
        self.lm_head = MaskedLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeForMaskedLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_attention_biases=output_attention_biases,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.lm_head(outputs, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMoeForMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeForPreTraining(RnaMoePreTrainedModel):

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        self.rnamoe = RnaMoeModel(config, add_pooling_layer=True)

        # Initialize weights and apply final processing
        self.post_init()


class RnaMoeForSecondaryStructurePrediction(RnaMoeForPreTraining):
    """
    Examples:
        >>> from multimolecule.models import RnaMoeConfig, RnaMoeForContactClassification, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForContactClassification(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        head_config = HeadConfig(num_layers=8)
        self.ss_head = ContactPredictionResNetHead(config, head_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeContactPredictorOutput:
        if output_attentions is False:
            warn("output_attentions must be True for contact classification and will be ignored.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.ss_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMoeContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeForDrugImprovement(RnaMoeForPreTraining):
    """
    Examples:
        >>> from multimolecule.models import RnaMoeConfig, RnaMoeForContactClassification, RnaTokenizer
        >>> config = RnaMoeConfig()
        >>> model = RnaMoeForContactClassification(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__(config)
        head_config = HeadConfig(transform="nonlinear")
        self.reactivity = TokenPredictionHead(config, head_config)
        self.deg_Mg_50C = TokenPredictionHead(config, head_config)
        self.deg_Mg_pH10 = TokenPredictionHead(config, head_config)
        self.deg_50C = TokenPredictionHead(config, head_config)
        self.deg_pH10 = TokenPredictionHead(config, head_config)
        self.proximal_isoform_proportion = SequencePredictionHead(config, head_config)
        self.crispr_on_target = SequencePredictionHead(config, head_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_reactivity: Tensor | None = None,
        labels_deg_Mg_50C: Tensor | None = None,
        labels_deg_Mg_pH10: Tensor | None = None,
        labels_deg_50C: Tensor | None = None,
        labels_deg_pH10: Tensor | None = None,
        labels_proximal_isoform_proportion: Tensor | None = None,
        labels_crispr_on_target: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMoeForDrugImprovementOutput:
        if output_attentions is False:
            warn("output_attentions must be True for contact classification and will be ignored.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamoe(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        output_reactivity = self.reactivity(outputs, attention_mask, input_ids, labels_reactivity)
        output_deg_Mg_50C = self.deg_Mg_50C(outputs, attention_mask, input_ids, labels_deg_Mg_50C)
        output_deg_Mg_pH10 = self.deg_Mg_pH10(outputs, attention_mask, input_ids, labels_deg_Mg_pH10)
        output_deg_50C = self.deg_50C(outputs, attention_mask, input_ids, labels_deg_50C)
        output_deg_pH10 = self.deg_pH10(outputs, attention_mask, input_ids, labels_deg_pH10)
        output_proximal_isoform_proportion = self.proximal_isoform_proportion(
            outputs, labels_proximal_isoform_proportion
        )
        output_crispr_on_target = self.crispr_on_target(outputs, labels_crispr_on_target)

        return RnaMoeForDrugImprovementOutput(
            loss_reactivity=output_reactivity.loss,
            logits_reactivity=output_reactivity.logits,
            loss_deg_Mg_50C=output_deg_Mg_50C.loss,
            logits_deg_Mg_50C=output_deg_Mg_50C.logits,
            loss_deg_Mg_pH10=output_deg_Mg_pH10.loss,
            logits_deg_Mg_pH10=output_deg_Mg_pH10.logits,
            loss_deg_50C=output_deg_50C.loss,
            logits_deg_50C=output_deg_50C.logits,
            loss_deg_pH10=output_deg_pH10.loss,
            logits_deg_pH10=output_deg_pH10.logits,
            loss_proximal_isoform_proportion=output_proximal_isoform_proportion.loss,
            logits_proximal_isoform_proportion=output_proximal_isoform_proportion.logits,
            loss_crispr_on_target=output_crispr_on_target.loss,
            logits_crispr_on_target=output_crispr_on_target.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaMoeEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embedding_type = getattr(config, "position_embedding_type", "sinusoidal")
        if self.position_embedding_type == "sinusoidal":
            self.position_embeddings = SinusoidalEmbedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id, bias=1
            )
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
            )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]  # type: ignore[union-attr]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embedding_type == "sinusoidal":
            position_embeddings = self.position_embeddings(input_ids)
        elif self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RnaMoeEncoder(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.config = config
        if config.num_moe_layers > config.num_hidden_layers:
            raise ValueError("num_moe_layers must be less than or equal to num_hidden_layers")
        if config.num_moe_layers <= 0 or config.num_experts <= 1:
            self.layer = nn.ModuleList([RnaMoeLayer(config) for _ in range(config.num_hidden_layers)])
        elif config.moe_layer_type == "intersperse":
            if config.num_hidden_layers % config.num_moe_layers != 0:
                raise ValueError("num_hidden_layers must be divisible by num_moe_layers")
            divider = config.num_hidden_layers // config.num_moe_layers
            self.layer = nn.ModuleList(
                [
                    RnaMoeSparseLayer(config) if i % divider == 0 else RnaMoeLayer(config)
                    for i in range(1, config.num_hidden_layers + 1)
                ]
            )
        elif config.moe_layer_type == "concat":
            layers = [RnaMoeLayer(config) for _ in range(config.num_hidden_layers - config.num_moe_layers)] + [
                RnaMoeSparseLayer(config) for _ in range(config.num_moe_layers)
            ]
            self.layer = nn.ModuleList(layers)
        else:
            raise ValueError("Invalid configuration for moe_layer_type")
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool = True,
        output_attention_biases: bool = True,
        output_hidden_states: bool = True,
        output_query: bool = True,
        output_key: bool = True,
        output_value: bool = True,
        output_router: bool = True,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | RnaMoeModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attention_biases = () if output_attention_biases else None
        all_self_attentions = () if output_attentions else None
        all_queries = () if output_query else None
        all_keys = () if output_key else None
        all_values = () if output_value else None
        all_router_probs = () if output_router else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    attention_bias,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    attention_bias,
                    layer_head_mask,
                )

            hidden_states, attention_bias = layer_outputs[:2]
            if output_attention_biases:
                all_attention_biases = all_attention_biases + (attention_bias,)  # type: ignore[operator]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)  # type: ignore[operator]
            if output_query:
                all_queries = all_queries + (layer_outputs[3],)  # type: ignore[operator]
            if output_key:
                all_keys = all_keys + (layer_outputs[4],)  # type: ignore[operator]
            if output_value:
                all_values = all_values + (layer_outputs[5],)  # type: ignore[operator]
            if output_router and len(layer_outputs) > 6:
                all_router_probs = all_router_probs + (layer_outputs[6],)  # type: ignore[operator]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_attention_biases,
                    all_queries,
                    all_keys,
                    all_values,
                    all_router_probs,
                ]
                if v is not None
            )
        return RnaMoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            attention_biases=all_attention_biases,
            attention_queries=all_queries,
            attention_keys=all_keys,
            attention_values=all_values,
            router_probs=all_router_probs,
        )


class RnaMoeLayer(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RnaMoeAttention(config)
        if config.ffn_type == "standard":
            self.ffn = RnaMoeFeedForward(config)
        elif config.ffn_type == "top2":
            self.ffn = RnaMoeTop2FeedForward(config)
        else:
            raise ValueError(f"ffn_type {config.ffn_type} not recognized.")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
    ) -> Tuple[Tensor, ...]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            attention_bias,
            head_mask,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]
        layer_output = self.ffn(attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class RnaMoeSparseLayer(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.expert_top_k
        self.top_p = config.expert_top_p
        self.attention = RnaMoeAttention(config)
        self.router = nn.Linear(config.hidden_size, self.num_experts)
        if config.ffn_type == "standard":
            layer = RnaMoeFeedForward
        elif config.ffn_type == "top2":
            layer = RnaMoeTop2FeedForward
        else:
            raise ValueError(f"ffn_type {config.ffn_type} not recognized.")
        self.experts = nn.ModuleList([layer(config) for _ in range(self.num_experts)])
        self._register_load_state_dict_pre_hook(self.copy_ffn_params)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
    ) -> Tuple[Tensor, ...]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            attention_bias,
            head_mask,
        )
        attention_output = self_attention_outputs[0]

        router_logits = self.router(attention_output[:, 0, :])
        router_probs = router_logits.softmax(dim=-1)
        router_weights, router_idx = router_probs.topk(self.top_k, dim=-1)
        router_weights[router_weights > self.top_p] = self.top_p

        router_weights /= router_weights.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([self.experts[i](attention_output) for i in range(self.num_experts)], dim=1)
        solicited_outputs = expert_outputs[torch.arange(router_idx.size(0)).unsqueeze(1), router_idx]
        layer_output = (solicited_outputs * router_weights.unsqueeze(-1).unsqueeze(-1)).sum(1)

        return (layer_output,) + self_attention_outputs[1:] + (router_probs,)

    def copy_ffn_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        ffn_prefix = prefix + "ffn."
        ffn_states = {k: v for k, v in state_dict.items() if k.startswith(ffn_prefix)}
        # ffn_states = {k: v for k, v in state_dict.items() if "layer_norm" not in k}
        for k, v in ffn_states.items():
            for i in range(self.num_experts):
                state_dict[k.replace("ffn.", f"experts.{i}.")] = v.clone()
            del state_dict[k]


class RnaMoeAttention(nn.Module):
    def __init__(self, config: RnaMoeConfig, position_embedding_type: str | None = None):
        super().__init__()
        self.self = RnaMoeSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RnaMoeSelfOutput(config)
        self.pruned_heads: set = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
    ) -> Tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            attention_bias,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RnaMoeSelfAttention(nn.Module):
    def __init__(self, config: RnaMoeConfig, position_embedding_type: str | None = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
    ) -> Tuple[Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore[attr-defined]

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RnaMoeModel forward() function)
            attention_scores = attention_scores + attention_mask
        if attention_bias is not None:
            attention_scores = attention_scores + attention_bias
            attention_bias = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, attention_bias, attention_probs, query_layer, key_layer, value_layer)


class RnaMoeSelfOutput(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RnaMoeTop2FeedForward(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.in_proj_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.in_proj_2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.activation(self.in_proj_1(hidden_states)) * self.in_proj_2(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class RnaMoeFeedForward(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + residual)
        return hidden_states


class RnaMoePooler(nn.Module):
    def __init__(self, config: RnaMoeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@dataclass
class RnaMoeSequencePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeTokenPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeContactPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeForMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeForDrugImprovementOutput(ModelOutput):
    loss_reactivity: torch.FloatTensor | None = None
    logits_reactivity: torch.FloatTensor = None
    loss_deg_Mg_50C: torch.FloatTensor | None = None
    logits_deg_Mg_50C: torch.FloatTensor = None
    loss_deg_Mg_pH10: torch.FloatTensor | None = None
    logits_deg_Mg_pH10: torch.FloatTensor = None
    loss_deg_50C: torch.FloatTensor | None = None
    logits_deg_50C: torch.FloatTensor = None
    loss_deg_pH10: torch.FloatTensor | None = None
    logits_deg_pH10: torch.FloatTensor = None
    loss_proximal_isoform_proportion: torch.FloatTensor | None = None
    logits_proximal_isoform_proportion: torch.FloatTensor = None
    loss_crispr_on_target: torch.FloatTensor | None = None
    logits_crispr_on_target: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None
    attention_queries: Tuple[torch.FloatTensor, ...] | None = None
    attention_keys: Tuple[torch.FloatTensor, ...] | None = None
    attention_values: Tuple[torch.FloatTensor, ...] | None = None
    router_probs: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMoeModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None
    attention_queries: Tuple[torch.FloatTensor, ...] | None = None
    attention_keys: Tuple[torch.FloatTensor, ...] | None = None
    attention_values: Tuple[torch.FloatTensor, ...] | None = None
    router_probs: Tuple[torch.FloatTensor, ...] | None = None


def get_pairwise_bias_dict(alpha):
    return FlatDict(
        {
            "AU": 2,
            "UA": 2,
            "CG": 3,
            "GC": 3,
            "GU": alpha,
            "UG": alpha,
        }
    )
