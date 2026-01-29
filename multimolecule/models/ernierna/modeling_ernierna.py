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
from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import ModelOutput
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
    SequencePredictionHead,
    SinusoidalEmbedding,
    TokenPredictionHead,
    eager_attention_forward,
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
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ErnieRnaLayer", "ErnieRnaEmbeddings"]


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
        self.gradient_checkpointing = False
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
        self._inited = False

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_pairwise_bias(
        self, input_ids: Tensor | NestedTensor, attention_mask: Tensor | NestedTensor | None = None
    ) -> Tensor | NestedTensor:
        batch_size, seq_length = input_ids.shape

        # Broadcasting data indices to compute indices
        data_index_x = input_ids.unsqueeze(2).expand(batch_size, seq_length, seq_length)
        data_index_y = input_ids.unsqueeze(1).expand(batch_size, seq_length, seq_length)

        # Get bias from pairwise_bias_map
        if not self._inited:
            token_to_ids = self.tokenizer._token_to_id
            tokens = sorted(token_to_ids, key=token_to_ids.get)
            pairwise_bias_dict = get_pairwise_bias_dict(self.config.pairwise_alpha)
            self.register_buffer(
                "pairwise_bias_map",
                torch.tensor(
                    [[pairwise_bias_dict.get(f"{i}{j}", 0) for i in tokens] for j in tokens], device=input_ids.device
                ),
                persistent=False,
            )
            self._inited = True
        return self.pairwise_bias_map[data_index_x, data_index_y]

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
        output_attention_biases: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
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
        output_attention_biases = (
            output_attention_biases if output_attention_biases is not None else self.config.output_attention_biases
        )
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

        pairwise_bias = self.get_pairwise_bias(input_ids, attention_mask)
        attention_bias = self.pairwise_bias_proj(pairwise_bias.unsqueeze(-1)).transpose(1, 3)

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
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attention_biases=output_attention_biases,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return ErnieRnaModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            attention_biases=encoder_outputs.attention_biases,
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

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ErnieRnaSequencePredictorOutput:
        outputs = self.ernierna(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

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

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ErnieRnaTokenPredictorOutput:
        outputs = self.ernierna(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

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

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ErnieRnaContactPredictorOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.ernierna(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

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

    _tied_weights_keys = {
        "lm_head.decoder.weight": "ernierna.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

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
    ) -> Tuple[Tensor, ...] | ErnieRnaForMaskedLMOutput:
        outputs = self.ernierna(
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
        >>> from multimolecule import ErnieRnaConfig, ErnieRnaForSecondaryStructurePrediction, RnaTokenizer
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

    @can_return_tuple
    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_lm: Tensor | None = None,
        labels_ss: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ErnieRnaForSecondaryStructurePredictorOutput:
        if self.require_attention_biases:
            output_attention_biases = kwargs.get("output_attention_biases", self.config.output_attention_biases)
            if output_attention_biases is False:
                warn("output_attention_biases must be True since prediction head requires attention biases.")
            kwargs["output_attention_biases"] = True
        outputs = self.ernierna(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output_lm = self.lm_head(outputs, labels_lm)
        logits_lm, loss_lm = output_lm.logits, output_lm.loss

        output_ss = self.ss_head(outputs, attention_mask, input_ids, labels_ss)
        logits_ss, loss_ss = output_ss.logits, output_ss.loss
        losses = tuple(l for l in (loss_lm, loss_ss) if l is not None)  # noqa: E741
        loss = torch.mean(torch.tensor(losses)) if losses else None

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
        self.layer = nn.ModuleList([ErnieRnaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        output_attention_biases: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | ErnieRnaModelOutputWithPastAndCrossAttentions:
        all_attention_biases = () if output_attention_biases else None
        for layer_module in self.layer:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                attention_bias,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states, attention_bias = layer_outputs[:2]
            if output_attention_biases:
                all_attention_biases = all_attention_biases + (attention_bias,)  # type: ignore[operator]

        return ErnieRnaModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attention_biases=all_attention_biases,
        )


class ErnieRnaLayer(GradientCheckpointingLayer):
    def __init__(self, config: ErnieRnaConfig, layer_idx: int | None = None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ErnieRnaAttention(config, layer_idx=layer_idx, is_causal=config.is_decoder)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieRnaAttention(
                config,
                position_embedding_type="absolute",
                layer_idx=layer_idx,
                is_causal=False,
                is_cross_attention=True,
            )
        self.intermediate = ErnieRnaIntermediate(config)
        self.output = ErnieRnaOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        self_attention_output, attention_bias, attn_weights = self.attention(
            hidden_states,
            attention_mask,
            attention_bias,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output
        outputs: tuple[Tensor, ...] = (attention_bias, attn_weights)

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attention_output, cross_attention_bias, cross_attn_weights = self.crossattention(
                attention_output,
                None,  # attention_mask
                None,  # attention_bias
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output
            outputs = outputs + (cross_attn_weights,)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return (layer_output,) + outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ErnieRnaAttention(nn.Module):
    def __init__(
        self,
        config: ErnieRnaConfig,
        position_embedding_type: str | None = None,
        layer_idx: int | None = None,
        is_causal: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = ErnieRnaCrossAttention if is_cross_attention else ErnieRnaSelfAttention
        self.self = attention_class(
            config,
            position_embedding_type=position_embedding_type,
            layer_idx=layer_idx,
            is_causal=is_causal,
        )
        self.output = ErnieRnaSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        if self.is_cross_attention:
            attention_output, attn_weights = self.self(
                hidden_states,
                encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_bias_out = attention_bias
        else:
            attention_output, attention_bias_out, attn_weights = self.self(
                hidden_states,
                attention_mask,
                attention_bias,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attention_bias_out, attn_weights


class ErnieRnaSelfAttention(nn.Module):
    def __init__(
        self,
        config: ErnieRnaConfig,
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
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

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
        attention_bias: torch.FloatTensor | None = None,
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

        attention_bias_mask = attention_mask
        relative_position_scores = None
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
            attention_bias_mask = (
                relative_position_scores
                if attention_bias_mask is None
                else attention_bias_mask + relative_position_scores
            )

        if attention_bias is not None:
            attention_bias_mask = (
                attention_bias if attention_bias_mask is None else attention_bias_mask + attention_bias
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_bias_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.all_head_size,)).contiguous()

        attention_bias_out = attention_bias
        if attention_bias is not None:
            attention_scores = (
                torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling  # type: ignore[attr-defined]
            )
            if relative_position_scores is not None:
                attention_scores = attention_scores + relative_position_scores
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_scores = attention_scores + attention_bias
            attention_bias_out = attention_scores

        return attn_output, attention_bias_out, attn_weights


class ErnieRnaCrossAttention(nn.Module):
    def __init__(
        self,
        config: ErnieRnaConfig,
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
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

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

        attention_bias = attention_mask
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

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


class ErnieRnaSelfOutput(nn.Module):
    def __init__(self, config: ErnieRnaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
            if isinstance(labels, NestedTensor) and not isinstance(output, NestedTensor):
                output = labels.nested_like(output, strict=False)
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


ErnieRnaPreTrainedModel._can_record_outputs = {
    "hidden_states": ErnieRnaLayer,
    "attentions": OutputRecorder(ErnieRnaAttention, index=2, layer_name="attention"),
    "cross_attentions": OutputRecorder(ErnieRnaAttention, index=2, layer_name="crossattention"),
}
