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
from .configuration_aido_rna import AidoRnaConfig, AidoRnaSecondaryStructureHeadConfig


class AidoRnaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AidoRnaConfig
    base_model_prefix = "aido_rna"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["AidoRnaLayer", "AidoRnaEmbeddings"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, AidoRnaEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))


class AidoRnaModel(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaModel, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 2048])
        >>> output["pooler_output"].shape
        torch.Size([1, 2048])
    """

    def __init__(self, config: AidoRnaConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = AidoRnaEmbeddings(config)
        self.encoder = AidoRnaEncoder(config)
        self.pooler = AidoRnaPooler(config) if add_pooling_layer else None

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

        if isinstance(input_ids, NestedTensor) and attention_mask is None:
            attention_mask = input_ids.mask
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


class AidoRnaForSequencePrediction(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForSequencePrediction, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AidoRnaConfig):
        super().__init__(config)
        self.aido_rna = AidoRnaModel(config)
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
        outputs = self.aido_rna(
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


class AidoRnaForTokenPrediction(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForTokenPrediction, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AidoRnaConfig):
        super().__init__(config)
        self.aido_rna = AidoRnaModel(config, add_pooling_layer=False)
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
        outputs = self.aido_rna(
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


class AidoRnaForContactPrediction(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForContactPrediction, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: AidoRnaConfig):
        super().__init__(config)
        self.aido_rna = AidoRnaModel(config, add_pooling_layer=False)
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
        outputs = self.aido_rna(
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


class AidoRnaForMaskedLM(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForMaskedLM, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """

    _tied_weights_keys = {
        "lm_head.decoder.weight": "aido_rna.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config: AidoRnaConfig):
        super().__init__(config)
        if config.is_decoder:
            warn(
                "If you want to use `AidoRnaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.aido_rna = AidoRnaModel(config, add_pooling_layer=False)
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
        outputs = self.aido_rna(
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


class AidoRnaForPreTraining(AidoRnaForMaskedLM):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForPreTraining, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForPreTraining(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=input["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 7, 26])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<NllLossBackward0>)
    """


class AidoRnaForSecondaryStructurePrediction(AidoRnaPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import AidoRnaConfig, AidoRnaForSecondaryStructurePrediction, RnaTokenizer
        >>> config = AidoRnaConfig()
        >>> model = AidoRnaForSecondaryStructurePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
    """

    def __init__(self, config: AidoRnaConfig):
        super().__init__(config)
        self.aido_rna = AidoRnaModel(config, add_pooling_layer=False)
        self.ss_head = AidoRnaSecondaryStructurePredictionHead(config)
        self.require_attentions = self.ss_head.require_attentions

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
        outputs = self.aido_rna(
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


class AidoRnaEmbeddings(nn.Module):
    def __init__(self, config: AidoRnaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

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
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]  # type: ignore[union-attr]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            if isinstance(embeddings, NestedTensor):
                if position_embeddings.size(0) == 1 and embeddings.tensor.size(0) != 1:
                    position_embeddings = position_embeddings.expand(embeddings.tensor.size(0), -1, -1)
                position_embeddings = embeddings.nested_like(position_embeddings, strict=False)
            embeddings = embeddings + position_embeddings

        if attention_mask is not None and not isinstance(embeddings, NestedTensor):
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings


class AidoRnaEncoder(nn.Module):
    def __init__(self, config: AidoRnaConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AidoRnaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
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


class AidoRnaLayer(GradientCheckpointingLayer):
    def __init__(self, config: AidoRnaConfig, layer_idx: int | None = None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AidoRnaAttention(config, layer_idx=layer_idx, is_causal=config.is_decoder)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = AidoRnaAttention(
                config,
                position_embedding_type="absolute",
                layer_idx=layer_idx,
                is_causal=False,
                is_cross_attention=True,
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = AidoRnaIntermediate(config)
        self.output = AidoRnaOutput(config)

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


class AidoRnaAttention(nn.Module):
    def __init__(
        self,
        config: AidoRnaConfig,
        position_embedding_type: str | None = None,
        layer_idx: int | None = None,
        is_causal: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = AidoRnaCrossAttention if is_cross_attention else AidoRnaSelfAttention
        self.self = attention_class(
            config,
            position_embedding_type=position_embedding_type,
            layer_idx=layer_idx,
            is_causal=is_causal,
        )
        self.output = AidoRnaSelfOutput(config)
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


class AidoRnaSelfAttention(nn.Module):
    def __init__(
        self,
        config: AidoRnaConfig,
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


class AidoRnaCrossAttention(nn.Module):
    def __init__(
        self,
        config: AidoRnaConfig,
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


class AidoRnaSelfOutput(nn.Module):
    def __init__(self, config: AidoRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AidoRnaIntermediate(nn.Module):
    def __init__(self, config: AidoRnaConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.gate(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = gate * hidden_states
        return hidden_states


class AidoRnaOutput(nn.Module):
    def __init__(self, config: AidoRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
class AidoRnaPooler(nn.Module):
    def __init__(self, config: AidoRnaConfig):
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


class AidoRnaSecondaryStructurePredictionHead(BasePredictionHead):

    config: AidoRnaSecondaryStructureHeadConfig
    output_name: str = "last_hidden_state"

    def __init__(self, config: AidoRnaConfig, head_config: AidoRnaSecondaryStructureHeadConfig | None = None):
        if head_config is None:
            head_config = (
                AidoRnaSecondaryStructureHeadConfig(config.head, num_labels=1)
                if config.head is not None
                else AidoRnaSecondaryStructureHeadConfig()
            )
        super().__init__(config, head_config)
        self.projection = nn.Linear(config.hidden_size * 2, self.config.num_channels)
        self.convnet = AidoRnaConvNet(self.config)
        self.prediction = nn.Conv2d(self.config.num_channels, 1, kernel_size=self.config.kernel_size, padding="same")
        self.criterion = Criterion(self.config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: BaseModelOutputWithPastAndCrossAttentions | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs[self.output_name]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")
        batch_size, _, _ = output.shape
        if batch_size > 1:
            warn(
                f"AIDO.RNA does not support `batch_size > 1`, but got {batch_size}.\n"
                "Using a larger batch size will result in incorrect predictions for all but the longest sequence due "
                "to how padding is handled.\n"
                "Please set `batch_size=1` for accurate results.",
                RuntimeWarning,
            )

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, _, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        contact_map = self.pairwise_concat(output)
        contact_map = self.projection(contact_map)
        contact_map = self.convnet(contact_map.permute(0, 3, 1, 2))
        contact_map = self.prediction(contact_map)
        triangular = contact_map.triu(diagonal=1)
        contact_map = triangular + triangular.transpose(-1, -2)
        contact_map = contact_map.permute(0, 2, 3, 1)

        if labels is not None:
            return HeadOutput(contact_map, self.criterion(contact_map, labels))
        return HeadOutput(contact_map)

    @staticmethod
    def pairwise_concat(hidden_states: Tensor) -> Tensor:
        """
        Creates pairwise concatenations of hidden states for all sequence positions.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            Tensor of shape (batch_size, seq_length, seq_length, hidden_size * 2)
            where output[b, i, j] has interleaved features from positions i and j
        """
        batch_size, seq_length, hidden_size = hidden_states.shape

        left = hidden_states.unsqueeze(2).expand(batch_size, seq_length, seq_length, hidden_size)
        right = hidden_states.unsqueeze(1).expand(batch_size, seq_length, seq_length, hidden_size)

        stacked = torch.stack((left, right), dim=-1)
        result = stacked.view(batch_size, seq_length, seq_length, hidden_size * 2)

        I, J = torch.tril_indices(seq_length, seq_length, -1, device=hidden_states.device)
        result[:, I, J] = result[:, J, I]

        return result


class AidoRnaConvNet(nn.Sequential):
    def __init__(self, config: AidoRnaSecondaryStructureHeadConfig):
        super().__init__(
            *[
                AidoRnaConvLayer(
                    config.num_channels, config.kernel_size, activation=config.activation, bias=config.bias
                )
                for _ in range(config.num_layers)
            ]
        )


class AidoRnaConvLayer(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, activation: str = "relu", bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=bias,
        )
        self.norm1 = nn.InstanceNorm2d(num_channels)
        self.activation1 = ACT2FN[activation]
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding="same",
        )
        self.norm2 = nn.InstanceNorm2d(num_channels)
        self.activation2 = ACT2FN[activation]
        self.conv3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=bias,
        )
        self.norm3 = nn.InstanceNorm2d(num_channels)
        self.activation3 = ACT2FN[activation]

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.conv3(hidden_state)
        hidden_state = self.norm3(hidden_state)
        hidden_state = self.activation3(hidden_state)
        return hidden_state + residual


AidoRnaPreTrainedModel._can_record_outputs = {
    "hidden_states": AidoRnaLayer,
    "attentions": OutputRecorder(AidoRnaAttention, index=1, layer_name="attention"),
    "cross_attentions": OutputRecorder(AidoRnaAttention, index=1, layer_name="crossattention"),
}
