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
from functools import partial
from typing import Tuple
from warnings import warn

import torch
from chanfig import ConfigRegistry
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from multimolecule.module import (
    ContactPredictionHead,
    MaskedLMHead,
    NucleotidePredictionHead,
    SequencePredictionHead,
    TokenPredictionHead,
)

from .configuration_rnamsm import RnaMsmConfig

LAYERS = ConfigRegistry()


class RnaMsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RnaMsmConfig
    base_model_prefix = "rnamsm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RnaMsmLayer", "RnaMsmAxialLayer", "RnaMsmPkmLayer", "RnaMsmEmbeddings"]

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
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RnaMsmModel(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmModel, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 768])
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    def __init__(self, config: RnaMsmConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = RnaMsmEmbeddings(config)
        self.encoder = RnaMsmEncoder(config)
        self.pooler = RnaMsmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmModelOutputWithPooling:
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = (
                input_ids.ne(self.pad_token_id) if self.pad_token_id is not None else torch.ones_like(input_ids)
            )

        unsqueeze_input = input_ids.ndim == 2
        if unsqueeze_input:
            input_ids = input_ids.unsqueeze(1)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if unsqueeze_input:
            sequence_output = sequence_output.squeeze(1)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return RnaMsmModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            col_attentions=encoder_outputs.col_attentions,
            row_attentions=encoder_outputs.row_attentions,
        )


class RnaMsmForContactPrediction(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForContactPrediction, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=True)
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
    ) -> Tuple[Tensor, ...] | RnaMsmContactPredictorOutput:
        if output_attentions is False:
            warn("output_attentions must be True for contact classification and will be ignored.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
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

        return RnaMsmContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmForNucleotidePrediction(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForNucleotidePrediction, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForNucleotidePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randn(1, 5, 2))
        >>> output["logits"].shape
        torch.Size([1, 5, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=True)
        self.nucleotide_head = NucleotidePredictionHead(config)
        self.head_config = self.nucleotide_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmNucleotidePredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.nucleotide_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMsmNucleotidePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmForSequencePrediction(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForSequencePrediction, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=True)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmSequencePredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMsmSequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmForTokenPrediction(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForTokenPrediction, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 7)))
        >>> output["logits"].shape
        torch.Size([1, 7, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=True)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmTokenPredictorOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMsmTokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmForMaskedLM(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForMaskedLM, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=False)
        self.lm_head = MaskedLMHead(config, weight=self.rnamsm.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmForMaskedLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = self.lm_head(outputs, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMsmForMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmForPreTraining(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForPreTraining, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForPreTraining(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels_mlm=input["input_ids"])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<AddBackward0>)
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["contact_map"].shape
        torch.Size([1, 5, 5, 2])
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=False)
        self.pretrain = RnaMsmPreTrainingHeads(config, weight=self.rnamsm.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_mlm: Tensor | None = None,
        labels_contact: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Tuple[Tensor, ...] | RnaMsmForPreTrainingOutput:
        if output_attentions is False:
            warn("output_attentions must be True for contact classification and will be ignored.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        total_loss, logits, contact_map = self.pretrain(
            outputs, attention_mask, input_ids, labels_mlm=labels_mlm, labels_contact=labels_contact
        )

        if not return_dict:
            output = (logits, contact_map) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return RnaMsmForPreTrainingOutput(
            loss=total_loss,
            logits=logits,
            contact_map=contact_map,
            hidden_states=outputs.hidden_states,
            col_attentions=outputs.col_attentions,
            row_attentions=outputs.row_attentions,
        )


class RnaMsmEmbeddings(nn.Module):
    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = RnaMsmLearnedPositionalEmbedding(
            self.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.register_buffer(
            "position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)) + 1, persistent=False
        )
        if config.embed_positions_msa:
            self.msa_embeddings = nn.Parameter(
                0.01 * torch.randn(1, self.max_position_embeddings, 1, 1), requires_grad=True
            )
        else:
            self.register_parameter("msa_embeddings", None)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]  # type: ignore[union-attr]
        _, num_alignments, seq_length = input_shape

        if attention_mask is None:
            if isinstance(input_ids, NestedTensor):
                input_ids, attention_mask = input_ids.tensor, input_ids.mask
            elif isinstance(inputs_embeds, NestedTensor):
                inputs_embeds, attention_mask = inputs_embeds.tensor, inputs_embeds.mask
            elif input_ids is not None and self.pad_token_id is not None:
                attention_mask = input_ids.ne(self.pad_token_id)
            else:
                raise ValueError("attention_mask is not passed and can not be inferred from input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length] * attention_mask.long()
        position_embeddings = self.position_embeddings(position_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.long())

        embeddings = inputs_embeds + position_embeddings

        if self.msa_embeddings is not None:
            if input_ids.size(1) > self.max_position_embeddings:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of {self.max_position_embeddings}, but received {num_alignments} alignments."
                )
            embeddings += self.msa_embeddings[:, :num_alignments]

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings


class RnaMsmLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, *args, **kwargs):
        num_embeddings += 2
        super().__init__(num_embeddings, *args, **kwargs)
        self.max_positions = num_embeddings

    def forward(self, position_ids: torch.LongTensor) -> Tensor:
        """Input is expected to be of size [bsz x seqlen]."""

        # This is a bug in the original implementation
        positions = position_ids + 1
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class RnaMsmEncoder(nn.Module):
    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RnaMsmAxialLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | RnaMsmModelOutput:
        key_padding_mask = (attention_mask ^ 1).bool()

        all_hidden_states = () if output_hidden_states else None
        all_col_attentions = () if output_attentions else None
        all_row_attentions = () if output_attentions else None

        # B x R x C x D -> R x C x B x D
        hidden_states = hidden_states.permute(1, 2, 0, 3)

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.permute(2, 0, 1, 3),)  # type: ignore[operator]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    key_padding_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    None,
                    key_padding_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                # H x C x B x R x R -> B x H x C x R x R
                all_col_attentions = all_col_attentions + (layer_outputs[1].permute(2, 0, 1, 3, 4),)  # type: ignore
                # H x B x C x C -> B x H x C x C
                all_row_attentions = all_row_attentions + (layer_outputs[2].permute(1, 0, 2, 3),)  # type: ignore

        # last hidden representation should have layer norm applied
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_col_attentions,
                    all_row_attentions,
                ]
                if v is not None
            )
        return RnaMsmModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            col_attentions=all_col_attentions,
            row_attentions=all_row_attentions,
        )


layer_registry = ConfigRegistry("layer_type")


@layer_registry.register("axial", default=True)
class RnaMsmAxialLayer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        row_self_attention = RowSelfAttention(config)
        column_self_attention = ColumnSelfAttention(config)
        feed_forward_layer = FeedForwardNetwork(config)

        self.row_self_attention = NormalizedResidualBlock(config, row_self_attention)
        self.column_self_attention = NormalizedResidualBlock(config, column_self_attention)
        self.feed_forward_layer = NormalizedResidualBlock(config, feed_forward_layer)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        row_attention_outputs = self.row_self_attention(
            hidden_states,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
        )
        row_attention_output, row_outputs = row_attention_outputs[0], row_attention_outputs[1:]
        col_attention_outputs = self.column_self_attention(
            row_attention_output,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
        )
        col_attention_output, col_outputs = col_attention_outputs[0], col_attention_outputs[1:]
        context_layer = self.feed_forward_layer(col_attention_output)

        outputs = (context_layer,) + col_outputs + row_outputs
        return outputs


@layer_registry.register("standard")
class RnaMsmLayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.self = attention_registry.build(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForwardNetwork(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attention_probs = self.self(
            hidden_states,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_probs


@layer_registry.register("pkm")
class RnaMsmPkmLayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, config: RnaMsmConfig):
        from product_key_memory import PKM

        super().__init__()
        self.self = attention_registry.build(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.pkm = PKM(
            config.hidden_size,
            config.pkm_attention_heads,
            config.num_product_keys,
            config.pkm_topk,
            config.pkm_head_size,
        )

        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attention_probs = self.self(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.pkm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_probs


class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.max_tokens_per_msa = config.max_tokens_per_msa
        self.attention_shape = "hnij"

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def compute_attention_weights(
        self,
        hidden_states,
        scaling: float,
        attention_mask=None,
        key_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(
            num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
        )
        k = self.k_proj(hidden_states).view(
            num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
        )
        q *= scaling
        if key_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - key_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q)
            # q *= 1 - key_padding_mask.permute(3, 4, 0, 1, 2).to(q)

        attention_scores = torch.einsum(f"rinhd,rjnhd->{self.attention_shape}", q, k)

        if attention_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(key_padding_mask[:, 0].unsqueeze(0).unsqueeze(2), -10000)

        return attention_scores

    def compute_attention_update(
        self,
        hidden_states,
        attention_probs,
    ):
        num_rows, num_cols, batch_size, hidden_size = hidden_states.size()
        v = self.v_proj(hidden_states).view(
            num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
        )
        context_layer = torch.einsum(f"{self.attention_shape},rjnhd->rinhd", attention_probs, v)
        context_layer = context_layer.reshape(num_rows, num_cols, batch_size, hidden_size)
        output = self.out_proj(context_layer)
        return output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        num_rows, num_cols, _, _ = hidden_states.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(hidden_states, attention_mask, key_padding_mask)
        scaling = self.align_scaling(hidden_states)
        attention_scores = self.compute_attention_weights(hidden_states, scaling, attention_mask, key_padding_mask)
        attention_probs = attention_scores.softmax(-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = self.compute_attention_update(hidden_states, attention_probs)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def _batched_forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ):
        num_rows, num_cols, _, _ = hidden_states.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        scaling = self.align_scaling(hidden_states)
        attention_scores = 0
        for start in range(0, num_rows, max_rows):
            attention_scores += self.compute_attention_weights(
                hidden_states[start : start + max_rows],
                scaling,
                attention_mask=attention_mask,
                key_padding_mask=(
                    key_padding_mask[:, start : start + max_rows] if key_padding_mask is not None else None
                ),
            )
        attention_probs = attention_scores.softmax(-1)  # type: ignore[attr-defined]
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.cat(
            [
                self.compute_attention_update(hidden_states[start : start + max_rows], attention_probs)
                for start in range(0, num_rows, max_rows)
            ],
            0,
        )
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ColumnSelfAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.max_tokens_per_msa = config.max_tokens_per_msa

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def compute_attention_update(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ):
        num_rows, num_cols, batch_size, hidden_size = hidden_states.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with
            # padding
            attention_probs = torch.ones(
                self.num_attention_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            context_layer = self.out_proj(self.v_proj(hidden_states))
        else:
            q = self.q_proj(hidden_states).view(
                num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
            )
            k = self.k_proj(hidden_states).view(
                num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
            )
            v = self.v_proj(hidden_states).view(
                num_rows, num_cols, batch_size, self.num_attention_heads, self.attention_head_size
            )
            q *= self.scaling

            attention_scores = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if attention_mask is not None:
                raise NotImplementedError
            if key_padding_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    key_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attention_probs = attention_scores.softmax(-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.einsum("hcnij,jcnhd->icnhd", attention_probs, v)
            context_layer = context_layer.reshape(num_rows, num_cols, batch_size, hidden_size)
            context_layer = self.out_proj(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        num_rows, num_cols, _, _ = hidden_states.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_cols) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                hidden_states,
                attention_mask,
                key_padding_mask,
            )
        else:
            return self.compute_attention_update(hidden_states, attention_mask, key_padding_mask, output_attentions)

    def _batched_forward(
        self,
        hidden_states,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
    ):
        num_rows, num_cols, _, _ = hidden_states.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        contexts, attentions = [], []
        for start in range(0, num_cols, max_cols):
            output, attention = self(
                hidden_states[:, start : start + max_cols],
                attention_mask=attention_mask,
                key_padding_mask=(
                    key_padding_mask[:, :, start : start + max_cols] if key_padding_mask is not None else None
                ),
            )
            contexts.append(output)
            attentions.append(attention)
        context_layer = torch.cat(contexts, 1)
        attention_probs = torch.cat(attentions, 1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


attention_registry = ConfigRegistry(key="attention_type")


@attention_registry.register("standard")
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.dropout_prob = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, config.attention_bias)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, config.attention_bias)

        self.reset_parameters()
        self.enable_torch_version = hasattr(F, "multi_head_attention_forward")
        if self.enable_torch_version:
            self._attention_fn = partial(
                F.multi_head_attention_forward,  # type: ignore
                embed_dim_to_check=self.hidden_size,
                num_heads=self.num_attention_heads,
                in_proj_weight=torch.empty([0]),
                bias_k=None,
                bias_v=None,
                add_zero_attention=False,
                dropout_p=self.dropout_prob,
                use_separate_proj_weight=True,
            )

    def attention_fn(
        self,
        query,
        key,
        value,
        key_padding_mask: Tensor | None = None,
        output_attentions: bool = False,
        attention_mask: Tensor | None = None,
    ):
        return self._attention_fn(
            query,
            key,
            value,
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            training=self.training,
            need_weights=output_attentions,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attention_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                pretrained and values before the attention softmax.
        """

        tgt_len, bsz, hidden_size = hidden_states.size()
        assert hidden_size == self.hidden_size

        # A workaround for quantization to work. Otherwise JIT compilation
        # treats bias in linear module as method.
        # and not output_attentions
        if self.enable_torch_version and not torch.jit.is_scripting():
            return self.attention_fn(
                query=hidden_states,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask,
                output_attentions=output_attentions,
                attention_mask=attention_mask,
            )

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q *= self.scaling

        q = q.reshape(tgt_len, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        k = k.reshape(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        v = v.reshape(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attention_scores = torch.bmm(q, k.transpose(1, 2))

        assert list(attention_scores.size()) == [bsz * self.num_attention_heads, tgt_len, src_len]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
            if self.onnx_trace:
                attention_mask = attention_mask.repeat(attention_scores.size(0), 1, 1)
            attention_scores += attention_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attention_scores = attention_scores.view(bsz, self.num_attention_heads, tgt_len, src_len)
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attention_scores = attention_scores.view(bsz * self.num_attention_heads, tgt_len, src_len)

        # attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        # attention_probs = attention_probs.type_as(attention_scores)
        attention_probs = attention_scores.softmax(-1)
        attention_probs = F.dropout(
            attention_probs.type_as(attention_scores),
            p=self.dropout_prob,
            training=self.training,
        )

        context_layer = torch.bmm(attention_probs, v)
        assert list(context_layer.size()) == [bsz * self.num_attention_heads, tgt_len, self.attention_head_size]
        context_layer = context_layer.transpose(0, 1).reshape(tgt_len, bsz, hidden_size)
        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


@attention_registry.register("performer")
class PerformerAttention(MultiheadAttention):
    def __init__(self, config: RnaMsmConfig):
        from performer_pytorch import FastAttention

        super().__init__(config)
        self._attention_fn = FastAttention(dim_heads=self.attention_head_size, nb_features=config.num_features)

    def attention_fn(self, query, key, value):
        return self._attention_fn(query, key, value)

    def forward(
        self,
        hidden_states: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        from einops import rearrange

        q = self.q_proj(hidden_states)  # [T x B x D]
        k = self.k_proj(hidden_states)  # [...]
        v = self.v_proj(hidden_states)  # [...]

        q, k, v = (rearrange(t, "t b (h d) -> b h t d", h=self.num_attention_heads) for t in (q, k, v))

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, :, None]
            v.masked_fill_(mask, 0)
        if attention_mask is not None:
            raise NotImplementedError

        attention_probs = self.attention_fn(q, k, v)
        context_layer = rearrange(attention_probs, "b h t d -> t b (h d)")
        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class NormalizedResidualBlock(nn.Module):
    def __init__(
        self,
        config: RnaMsmConfig,
        layer: nn.Module,
    ):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(config.attention_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, *args, **kwargs) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = self.layer(hidden_states, *args, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states, *out = outputs
        else:
            hidden_states = outputs
            out = None

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        if out is not None:
            return (hidden_states,) + tuple(out)
        else:
            return hidden_states


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: RnaMsmConfig):
        super().__init__()
        self.activation = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states) -> Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class RnaMsmPooler(nn.Module):
    def __init__(self, config: RnaMsmConfig):
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


class RnaMsmPreTrainingHeads(nn.Module):
    def __init__(self, config: RnaMsmConfig, weight: Tensor | None = None):
        super().__init__()
        self.predictions = MaskedLMHead(config, weight=weight)
        self.contact_head = ContactPredictionHead(config)

    def forward(
        self,
        outputs: RnaMsmModelOutput | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | NestedTensor | None = None,
        labels_mlm: Tensor | None = None,
        labels_contact: Tensor | None = None,
    ) -> Tuple[Tensor | None, Tensor, Tensor]:
        output_mlm = self.predictions(outputs, labels=labels_mlm)
        output_contact = self.contact_head(outputs, attention_mask, input_ids, labels=labels_contact)

        losses = [output.loss for output in (output_mlm, output_contact) if output.loss is not None]
        total_loss = sum(losses) if losses else None

        return total_loss, output_mlm.logits, output_contact.logits


@dataclass
class RnaMsmContactPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmNucleotidePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmSequencePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmTokenPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmForMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    contact_map: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    row_attentions: Tuple[torch.FloatTensor, ...] | None = None
