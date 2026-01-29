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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Tuple
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, check_model_inputs

from multimolecule.modules import (
    BasePredictionHead,
    ContactAttentionHead,
    ContactPredictionHead,
    Criterion,
    HeadOutput,
    MaskedLMHead,
    RotaryEmbedding,
    SequencePredictionHead,
    TokenPredictionHead,
    eager_attention_forward,
)

from ..modeling_outputs import ContactPredictorOutput, SequencePredictorOutput, TokenPredictorOutput
from .configuration_rnafm import RnaFmConfig, RnaFmSecondaryStructureHeadConfig


class RnaFmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RnaFmConfig
    base_model_prefix = "rnafm"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["RnaFmLayer", "RnaFmEmbeddings"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, RnaFmEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


class RnaFmModel(RnaFmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmModel, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 640])
        >>> output["pooler_output"].shape
        torch.Size([1, 640])
    """

    def __init__(self, config: RnaFmConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = RnaFmEmbeddings(config)
        self.encoder = RnaFmEncoder(config)
        self.pooler = RnaFmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @check_model_inputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
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
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = inputs_embeds.device  # type: ignore[union-attr]
            seq_length = inputs_embeds.shape[1]  # type: ignore[union-attr]

        # past_key_values_length
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        if attention_mask is None and input_ids is not None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        cache_position,
        past_key_values,
    ):
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config, input_embeds=embedding_output, attention_mask=attention_mask
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


class RnaFmForSequencePrediction(RnaFmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForSequencePrediction, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.rnafm = RnaFmModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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


class RnaFmForTokenPrediction(RnaFmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForTokenPrediction, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.rnafm = RnaFmModel(config, add_pooling_layer=False)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | TokenPredictorOutput:
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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


class RnaFmForContactPrediction(RnaFmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForContactPrediction, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.rnafm = RnaFmModel(config, add_pooling_layer=False)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        self.require_attentions = self.contact_head.require_attentions

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ContactPredictorOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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


class RnaFmForMaskedLM(RnaFmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForMaskedLM, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    _tied_weights_keys = {
        "lm_head.decoder.weight": "rnafm.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        if config.is_decoder:
            warn(
                "If you want to use `RnaFmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.rnafm = RnaFmModel(config, add_pooling_layer=False)
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
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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


class RnaFmForPreTraining(RnaFmForMaskedLM):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForPreTraining, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForPreTraining(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels_lm=input["input_ids"])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MeanBackward0>)
        >>> output["logits_lm"].shape
        torch.Size([1, 7, 26])
        >>> output["logits_ss"].shape
        torch.Size([1, 5, 5, 1])
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.ss_head = ContactAttentionHead(config)
        self.require_attentions = self.ss_head.require_attentions

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels_lm: Tensor | None = None,
        labels_ss: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RnaFmForPreTrainingOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

        output_lm = self.lm_head(outputs, labels=labels_lm)
        logits_lm, loss_lm = output_lm.logits, output_lm.loss

        output_ss = self.ss_head(outputs, attention_mask, input_ids, labels=labels_ss)
        logits_ss, loss_ss = output_ss.logits, output_ss.loss

        losses = tuple(l for l in (loss_lm, loss_ss) if l is not None)  # noqa: E741
        loss = torch.mean(torch.stack(losses)) if losses else None

        return RnaFmForPreTrainingOutput(
            loss=loss,
            logits_lm=logits_lm,
            loss_lm=loss_lm,
            logits_ss=logits_ss,
            loss_ss=loss_ss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaFmForSecondaryStructurePrediction(RnaFmForPreTraining):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RnaFmConfig, RnaFmForSecondaryStructurePrediction, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForSecondaryStructurePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.rnafm = RnaFmModel(config, add_pooling_layer=False)
        self.ss_head = RnaFmSecondaryStructurePredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ContactPredictorOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            **kwargs,
        )

        output = self.ss_head(outputs, attention_mask, input_ids, labels=labels)
        logits, loss = output.logits, output.loss

        return ContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaFmEmbeddings(nn.Module):
    def __init__(self, config: RnaFmConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.embed_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
        else:
            self.position_embeddings = None
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        past_key_values_length: int = 0,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.token_dropout:
            if input_ids is None:
                raise ValueError("Token dropout is only supported when input_ids are provided")
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all RNA-FM model training runs
            src_lengths = attention_mask.sum(-1)  # type: ignore[union-attr]
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(embeddings)

        if self.position_embedding_type == "absolute":
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(
                        input_ids, self.padding_idx, past_key_values_length
                    )
                else:
                    position_ids = create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)
                # This is a bug in the original implementation
                position_ids = position_ids + 1
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings


class RnaFmEncoder(nn.Module):
    def __init__(self, config: RnaFmConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RnaFmLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class RnaFmLayer(GradientCheckpointingLayer):
    def __init__(self, config: RnaFmConfig, layer_idx: int | None = None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RnaFmAttention(config, layer_idx=layer_idx, is_causal=config.is_decoder)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RnaFmAttention(
                config,
                position_embedding_type="absolute",
                layer_idx=layer_idx,
                is_causal=False,
                is_cross_attention=True,
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = RnaFmIntermediate(config)
        self.output = RnaFmOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attention_output, _ = self.crossattention(
                attention_output,
                None,  # attention_mask
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.layer_norm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RnaFmAttention(nn.Module):
    def __init__(
        self,
        config: RnaFmConfig,
        position_embedding_type: str | None = None,
        layer_idx: int | None = None,
        is_causal: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = RnaFmCrossAttention if is_cross_attention else RnaFmSelfAttention
        self.self = attention_class(
            config,
            position_embedding_type=position_embedding_type,
            layer_idx=layer_idx,
            is_causal=is_causal,
        )
        self.output = RnaFmSelfOutput(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        hidden_states_ln = self.layer_norm(hidden_states)
        if self.is_cross_attention:
            attention_mask = encoder_attention_mask
            attention_output, attn_weights = self.self(
                hidden_states_ln,
                encoder_hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            attention_output, attn_weights = self.self(
                hidden_states_ln,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class RnaFmSelfAttention(nn.Module):
    def __init__(
        self,
        config: RnaFmConfig,
        position_embedding_type: str | None = None,
        layer_idx: int | None = None,
        is_causal: bool | None = None,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(embedding_dim=self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.is_causal = config.is_decoder if is_causal is None else is_causal
        self.layer_idx = layer_idx

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError("layer_idx must be set when using past_key_values.")

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if past_key_values is not None:
            current_past_key_values = (
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values
            )
            key_layer, value_layer = current_past_key_values.update(
                key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)  # type: ignore[misc]

        attention_bias = attention_mask
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            else:
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                relative_position_scores = relative_position_scores_query + relative_position_scores_key

            relative_position_scores = relative_position_scores * self.scaling
            attention_bias = (
                relative_position_scores if attention_bias is None else attention_bias + relative_position_scores
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_bias,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.all_head_size,)).contiguous()
        return attn_output, attn_weights


class RnaFmCrossAttention(nn.Module):
    def __init__(
        self,
        config: RnaFmConfig,
        position_embedding_type: str | None = None,
        layer_idx: int | None = None,
        is_causal: bool | None = None,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(embedding_dim=self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.is_causal = config.is_decoder if is_causal is None else is_causal
        self.layer_idx = layer_idx

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided for cross-attention.")
        mixed_query_layer = self.query(hidden_states)
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError("layer_idx must be set when using past_key_values.")

        if past_key_values is not None:
            if not isinstance(past_key_values, EncoderDecoderCache):
                raise ValueError("Cross-attention caching requires EncoderDecoderCache.")
            is_updated = past_key_values.is_updated.get(self.layer_idx) if self.layer_idx is not None else False
            if is_updated:
                key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
                value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
            else:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True
        else:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)  # type: ignore[misc]

        attention_bias = attention_mask
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            else:
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                relative_position_scores = relative_position_scores_query + relative_position_scores_key

            relative_position_scores = relative_position_scores * self.scaling
            attention_bias = (
                relative_position_scores if attention_bias is None else attention_bias + relative_position_scores
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_bias,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.all_head_size,)).contiguous()
        return attn_output, attn_weights


class RnaFmSelfOutput(nn.Module):
    def __init__(self, config: RnaFmConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class RnaFmIntermediate(nn.Module):
    def __init__(self, config: RnaFmConfig):
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


class RnaFmOutput(nn.Module):
    def __init__(self, config: RnaFmConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RnaFmPooler(nn.Module):
    def __init__(self, config: RnaFmConfig):
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


class RnaFmSecondaryStructurePredictionHead(BasePredictionHead):

    config: RnaFmSecondaryStructureHeadConfig
    require_attentions: bool = False
    output_name: str = "last_hidden_state"

    def __init__(self, config: RnaFmConfig, head_config: RnaFmSecondaryStructureHeadConfig | None = None):
        if head_config is None:
            head_config = (
                RnaFmSecondaryStructureHeadConfig(config.head, num_labels=1)
                if config.head is not None
                else RnaFmSecondaryStructureHeadConfig()
            )
        super().__init__(config, head_config)
        if self.config.hidden_size is not None:
            self.reduction = nn.Linear(config.hidden_size, self.config.hidden_size)
            self.hidden_size = self.config.hidden_size
        else:
            self.reduction = nn.Identity()
            self.hidden_size = config.hidden_size
        self.projection = nn.Conv2d(self.hidden_size * 2, self.config.num_channels, kernel_size=1)
        self.convnet = RnaFmConvNet(self.config)
        self.prediction = nn.Conv2d(self.config.num_channels, self.config.num_labels, kernel_size=3, padding=1)
        self.criterion = Criterion(self.config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[-1]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, _, _ = self.remove_special_tokens(output, attention_mask, input_ids)
        seq_length = output.size(1)

        output = self.reduction(output)
        contact_map = output.unsqueeze(2).expand(-1, -1, seq_length, -1)
        contact_map = torch.cat([contact_map, contact_map.transpose(1, 2)], dim=-1)
        contact_map = self.projection(contact_map.permute(0, 3, 1, 2))
        contact_map = self.convnet(contact_map)
        contact_map = self.prediction(contact_map)
        contact_map = contact_map.triu() + contact_map.triu(diagonal=1).transpose(-1, -2)
        contact_map = contact_map.permute(0, 2, 3, 1)

        if labels is not None:
            return HeadOutput(contact_map, self.criterion(contact_map, labels))
        return HeadOutput(contact_map)


class RnaFmConvNet(nn.Sequential):
    def __init__(self, config: RnaFmSecondaryStructureHeadConfig) -> None:
        layers = []
        for i in range(config.num_layers):
            layers.append(
                RnaFmConvBlock(
                    config.num_channels,
                    dilation=pow(2, (i % 3)),
                    dropout=config.dropout,
                    activation=config.activation,
                    bias=config.bias,
                )
            )
        super().__init__(*layers)


class RnaFmConvBlock(nn.Module):
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
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.activation = ACT2FN[activation]
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size, stride=stride, padding=1, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, dilation=dilation, padding=dilation, bias=bias)

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.bn1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


@dataclass
class RnaFmForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_lm: torch.FloatTensor = None
    loss_lm: torch.FloatTensor | None = None
    logits_ss: torch.FloatTensor = None
    loss_ss: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaFmForSecondaryStructurePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_lm: torch.FloatTensor = None
    loss_lm: torch.FloatTensor = None
    logits_ss: torch.FloatTensor = None
    loss_ss: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
    attention_biases: Tuple[torch.FloatTensor, ...] | None = None


def create_position_ids_from_inputs_embeds(inputs_embeds: torch.FloatTensor, padding_idx: int = 0) -> torch.LongTensor:
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(
        padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(input_shape)


def create_position_ids_from_input_ids(
    input_ids: torch.LongTensor, padding_idx: int = 0, past_key_values_length: int = 0
) -> torch.LongTensor:
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        (torch.cumsum(mask, dim=1, dtype=mask.dtype) + past_key_values_length) * mask + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


RnaFmPreTrainedModel._can_record_outputs = {
    "hidden_states": RnaFmLayer,
    "attentions": OutputRecorder(RnaFmAttention, index=1, layer_name="attention"),
    "cross_attentions": OutputRecorder(RnaFmAttention, index=1, layer_name="crossattention"),
}
