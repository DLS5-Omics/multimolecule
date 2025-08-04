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
from collections.abc import Mapping
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
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from multimolecule.modules import (
    BasePredictionHead,
    ContactPredictionHead,
    Criterion,
    HeadOutput,
    MaskedLMHead,
    SequencePredictionHead,
    SinusoidalEmbedding,
    TokenPredictionHead,
)

from .configuration_ernierna import ErnieRnaConfig, ErnieRnaSecondaryStructureHeadConfig


class ErnieRnaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ErnieRnaConfig
    base_model_prefix = "ernierna"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ErnieRnaLayer", "ErnieRnaEmbeddings"]

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


class ErnieRnaModel(ErnieRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaModel, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaModel(config)
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
        self, config: ErnieRnaConfig, add_pooling_layer: bool = True, tokenizer: PreTrainedTokenizer | None = None
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
        self.embeddings = ErnieRnaEmbeddings(config)
        self.encoder = ErnieRnaEncoder(config)
        self.pooler = ErnieRnaPooler(config) if add_pooling_layer else None

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
        batch_size, seq_length = input_ids.shape

        # Broadcasting data indices to compute indices
        data_index_x = input_ids.unsqueeze(2).expand(batch_size, seq_length, seq_length)
        data_index_y = input_ids.unsqueeze(1).expand(batch_size, seq_length, seq_length)

        # Get bias from pairwise_bias_map
        return self.pairwise_bias_map[data_index_x, data_index_y]

        # Zhiyuan: Is it really necessary to mask the bias?
        # The mask position should have been nan, and the implementation is incorrect anyway
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)
        #     bias = bias * attention_mask

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | ErnieRnaModelOutputWithPoolingAndCrossAttentions:
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
            use_cache:
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if kwargs:
            warn(
                f"Additional keyword arguments `{', '.join(kwargs)}` are detected in "
                f"`{self.__class__.__name__}.forward`, they will be ignored.\n"
                "This is provided for backward compatibility and may lead to unexpected behavior."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attention_biases = (
            output_attention_biases if output_attention_biases is not None else self.config.output_attention_biases
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

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

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            if input_ids is not None and self.pad_token_id is not None:
                attention_mask = input_ids.ne(self.pad_token_id)
            else:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
                warn(
                    "attention_mask is not specified, and cannot be inferred from input_ids."
                    "Assuming all tokens are not masked."
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

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
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            attention_bias=attention_bias,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_attention_biases=output_attention_biases,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return ErnieRnaModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attention_biases=encoder_outputs.attention_biases,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ErnieRnaForSequencePrediction(ErnieRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaForSequencePrediction, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        self.ernierna = ErnieRnaModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
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
    ) -> Tuple[Tensor, ...] | ErnieRnaSequencePredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernierna(
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

        return ErnieRnaSequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieRnaForTokenPrediction(ErnieRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaForTokenPrediction, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ernierna = ErnieRnaModel(config, add_pooling_layer=False)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
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
    ) -> Tuple[Tensor, ...] | ErnieRnaTokenPredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernierna(
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

        return ErnieRnaTokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieRnaForContactPrediction(ErnieRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaForContactPrediction, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        self.ernierna = ErnieRnaModel(config, add_pooling_layer=False)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        self.require_attentions = self.contact_head.require_attentions

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | ErnieRnaContactPredictorOutput:
        if self.require_attentions:
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            output_attentions = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernierna(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ErnieRnaContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieRnaForMaskedLM(ErnieRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaForMaskedLM, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        if config.is_decoder:
            warn(
                "If you want to use `ErnieRnaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.ernierna = ErnieRnaModel(config, add_pooling_layer=False)
        self.lm_head = MaskedLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
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
    ) -> Tuple[Tensor, ...] | ErnieRnaForMaskedLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernierna(
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

        return ErnieRnaForMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieRnaForPreTraining(ErnieRnaForMaskedLM):

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        self.ernierna = ErnieRnaModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class ErnieRnaForSecondaryStructurePrediction(ErnieRnaForPreTraining):
    """
    Examples:
        >>> import torch
        >>> from multimolecule.models import ErnieRnaConfig, ErnieRnaForSecondaryStructurePrediction, RnaTokenizer
        >>> config = ErnieRnaConfig()
        >>> model = ErnieRnaForSecondaryStructurePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits_ss"].shape
        torch.Size([1, 5, 5, 1])
    """

    def __init__(self, config: ErnieRnaConfig):
        super().__init__(config)
        self.ss_head = ErnieRnaSecondaryStructurePredictionHead(config)
        self.require_attention_biases = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_lm: Tensor | None = None,
        labels_ss: Tensor | None = None,
        output_attentions: bool | None = None,
        output_attention_biases: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | ErnieRnaForSecondaryStructurePredictorOutput:
        if self.require_attention_biases:
            if output_attention_biases is False:
                warn("output_attention_biases must be True since prediction head requires attention biases.")
            output_attention_biases = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernierna(
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

        output_lm = self.lm_head(outputs, labels_lm)
        logits_lm, loss_lm = output_lm.logits, output_lm.loss

        output_ss = self.ss_head(outputs, attention_mask, input_ids, labels_ss)
        logits_ss, loss_ss = output_ss.logits, output_ss.loss
        losses = tuple(l for l in (loss_lm, loss_ss) if l is not None)  # noqa: E741
        loss = torch.mean(torch.tensor(losses)) if losses else None

        if not return_dict:
            output = outputs[2:]
            output = ((logits_ss, loss_ss) + output) if loss_ss is not None else ((logits_ss,) + output)
            output = ((logits_lm, loss_lm) + output) if loss_lm is not None else ((logits_lm,) + output)
            return ((loss,) + output) if loss is not None else output

        return ErnieRnaForSecondaryStructurePredictorOutput(
            loss=loss,
            logits_lm=logits_lm,
            loss_lm=loss_lm,
            logits_ss=logits_ss,
            loss_ss=loss_ss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_biases=outputs.attention_biases,
        )


class ErnieRnaEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: ErnieRnaConfig):
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

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "sinusoidal":
            position_embeddings = self.position_embeddings(input_ids)
            embeddings += position_embeddings
        elif self.position_embedding_type == "absolute":
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieRnaEncoder(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ErnieRnaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor, ...], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_attention_biases: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | ErnieRnaModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_attention_biases = () if output_attention_biases else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training and use_cache:
            warn("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    attention_bias,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    attention_bias,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states, attention_bias = layer_outputs[:2]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)  # type: ignore[operator]
            if output_attention_biases:
                all_attention_biases = all_attention_biases + (attention_bias,)  # type: ignore[operator]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)  # type: ignore[operator]
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[3],)  # type: ignore[operator]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                    all_attention_biases,
                ]
                if v is not None
            )
        return ErnieRnaModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,  # type: ignore[arg-type]
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            attention_biases=all_attention_biases,
        )


class ErnieRnaLayer(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ErnieRnaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieRnaAttention(config, position_embedding_type="absolute")
        self.intermediate = ErnieRnaIntermediate(config)
        self.output = ErnieRnaOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            attention_bias,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ErnieRnaAttention(nn.Module):
    def __init__(self, config: ErnieRnaConfig, position_embedding_type: str | None = None):
        super().__init__()
        self.self = ErnieRnaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ErnieRnaSelfOutput(config)
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
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            attention_bias,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ErnieRnaSelfAttention(nn.Module):
    def __init__(self, config: ErnieRnaConfig, position_embedding_type: str | None = None):
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

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
        output_attentions: bool = False,
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

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore[attr-defined]

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
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
            # Apply the attention mask is (precomputed for all layers in ErnieRnaModel forward() function)
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

        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs: Tuple[Tensor, ...] = (context_layer, attention_bias)

        if output_attentions:
            outputs = outputs + (attention_probs,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class ErnieRnaSelfOutput(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class ErnieRnaIntermediate(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Ernie
class ErnieRnaOutput(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
class ErnieRnaPooler(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
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


class ErnieRnaSecondaryStructurePredictionHead(BasePredictionHead):

    config: ErnieRnaSecondaryStructureHeadConfig
    output_name: str = "attention_biases"
    require_attention_biases: bool = True

    def __init__(self, config: ErnieRnaConfig, head_config: ErnieRnaSecondaryStructureHeadConfig | None = None):
        if head_config is None:
            head_config = (
                ErnieRnaSecondaryStructureHeadConfig(config.head, num_labels=1)
                if config.head is not None
                else ErnieRnaSecondaryStructureHeadConfig()
            )
        super().__init__(config, head_config)
        intermediate_channels = round(self.config.num_channels**0.5)
        self.activation = ACT2FN[self.config.activation]
        self.conv1 = nn.Conv2d(1, intermediate_channels, self.config.kernel_size, padding=self.config.kernel_size // 2)
        self.dropout = nn.Dropout(self.config.dropout)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            self.config.num_channels - 1,
            self.config.kernel_size,
            padding=self.config.kernel_size // 2,
        )
        self.convnet = ErnieRnaConvNet(self.config)
        self.soft_sign = ErnieRnaSoftsign()
        self.criterion = Criterion(self.config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ErnieRnaModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
        num_iters: int = 100,
        lr_min: float = 0.01,
        lr_max: float = 0.1,
        sparsity: float = 1.6,
        threshold: float = 1.5,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            attentions = outputs[self.output_name]
        elif isinstance(outputs, tuple):
            attentions = outputs[-1]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        attention = attentions[-1][:, 5:6, :, :].permute(0, 2, 3, 1)  # Mysterious magic head 5

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        attention, _, input_ids = self.remove_special_tokens_2d(attention, attention_mask, input_ids)

        attention = attention.permute(0, 3, 1, 2)
        output = self.conv1(attention)
        output = self.dropout(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = torch.cat((output, attention), dim=1)
        output = self.convnet(output)

        output = (output + output.transpose(2, 3)).squeeze(1)

        constraint = self._get_canonical_pair_constraint_matrix(input_ids)
        output = self.soft_sign(output - threshold) * output
        adjacency = output.sigmoid() * self.soft_sign(output - threshold).detach()

        matrix_adj = self._get_squared_adjacency_matrix(adjacency, constraint)
        lambda_values = self.activation(torch.sum(matrix_adj, dim=-1) - 1).detach()

        for _ in range(num_iters):
            matrix_adj = self._get_squared_adjacency_matrix(adjacency, constraint)
            grad_adjacency = lambda_values * self.soft_sign(torch.sum(matrix_adj, dim=-1) - 1)
            grad_adjacency = grad_adjacency.unsqueeze(-1).expand(output.shape) - output / 2
            gradient = adjacency * constraint * (grad_adjacency + grad_adjacency.transpose(-1, -2))

            adjacency -= lr_min * gradient
            lr_min = lr_min * 0.99
            adjacency = self.activation(torch.abs(adjacency) - sparsity * lr_min)

            matrix_adj = self._get_squared_adjacency_matrix(adjacency, constraint)
            lambda_gradient = self.activation(torch.sum(matrix_adj, dim=-1) - 1)
            lambda_values += lr_max * lambda_gradient
            lr_max = lr_max * 0.99

        output = adjacency * adjacency
        output = (output + output.transpose(-1, -2)) / 2
        output = output * constraint
        output = output.unsqueeze(-1)

        if labels is not None:
            if isinstance(labels, NestedTensor):
                if isinstance(output, Tensor):
                    output = labels.nested_like(output, strict=False)
                return HeadOutput(output, self.criterion(output.concat, labels.concat))
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)

    @staticmethod
    def _get_canonical_pair_constraint_matrix(input_ids: Tensor) -> Tensor:
        """
        Computes a constraint matrix for RNA base pairing.

        This matrix indicates which positions can potentially form base pairs based on
        Watson-Crick and wobble base pairing rules (A-U, C-G, G-U).

        Args:
            input_ids: Token IDs where 6=A, 7=C, 8=G, 9=U

        Returns:
            A binary matrix where 1 indicates positions that can form valid base pairs
        """
        dtype = torch.get_default_dtype()
        base_a = (input_ids == 6).to(dtype)
        base_c = (input_ids == 7).to(dtype)
        base_g = (input_ids == 8).to(dtype)
        base_u = (input_ids == 9).to(dtype)
        batch_size, seq_length = input_ids.shape

        au = torch.matmul(base_a.view(batch_size, seq_length, 1), base_u.view(batch_size, 1, seq_length))
        cg = torch.matmul(base_c.view(batch_size, seq_length, 1), base_g.view(batch_size, 1, seq_length))
        gu = torch.matmul(base_g.view(batch_size, seq_length, 1), base_u.view(batch_size, 1, seq_length))

        return au + au.transpose(1, 2) + cg + cg.transpose(1, 2) + gu + gu.transpose(1, 2)

    @staticmethod
    def _get_squared_adjacency_matrix(adjacency, constraint):
        """
        Computes the squared adjacency matrix with symmetry constraints.

        Args:
            adjacency: Estimated adjacency values
            constraint: Matrix of valid base pairings

        Returns:
            Symmetrized squared adjacency matrix
        """
        adjacency = adjacency * adjacency
        adjacency = (adjacency + torch.transpose(adjacency, -1, -2)) / 2
        return adjacency * constraint


class ErnieRnaConvNet(nn.Sequential):
    def __init__(self, config: ErnieRnaSecondaryStructureHeadConfig) -> None:
        layers = []
        for i in range(config.num_layers):
            layers.append(
                ErnieRnaConvBlock(
                    config.num_channels,
                    dilation=pow(2, (i % 3)),
                    dropout=config.dropout,
                    activation=config.activation,
                    bias=config.bias,
                )
            )
        layers.append(nn.Conv2d(config.num_channels, config.num_labels, kernel_size=3, padding=1))
        super().__init__(*layers)


class ErnieRnaConvBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.3,
        activation: str = "relu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_channels)
        self.activation = ACT2FN[activation]
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=1, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


@dataclass
class ErnieRnaModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    past_key_values: Tuple[Tuple[torch.FloatTensor, ...]] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Tuple[Tuple[torch.FloatTensor, ...]] | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaSequencePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaTokenPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaContactPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaForMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ErnieRnaForSecondaryStructurePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_lm: torch.FloatTensor = None
    loss_lm: torch.FloatTensor = None
    logits_ss: torch.FloatTensor = None
    loss_ss: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


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


class ErnieRnaSoftsign(nn.Module):

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return 1.0 / (1.0 + torch.exp(-2 * self.k * input))
