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
from transformers.modeling_utils import OutputRecorder, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import TokenPredictionHead

from .configuration_sptransformer import SpTransformerConfig


class SpTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["SpTransformerLayer", "SpTransformerResidualBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class SpTransformerModel(SpTransformerPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import SpTransformerConfig, SpTransformerFeatureEncoderConfig, SpTransformerModel
        >>> encoder = SpTransformerFeatureEncoderConfig(hidden_size=4)
        >>> config = SpTransformerConfig(
        ...     context=2, hidden_size=8, encoders=[encoder], attention_hidden_size=16,
        ...     num_hidden_layers=1, num_attention_heads=2, num_local_attention_heads=1,
        ...     intermediate_size=32, bucket_size=4, max_seq_len=8, num_tissues=2,
        ... )
        >>> model = SpTransformerModel(config)
        >>> input_ids = torch.randint(5, (1, 8))
        >>> output = model(input_ids)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 8, 16])
        >>> output["logits"].shape
        torch.Size([1, 8, 5])
    """

    def __init__(self, config: SpTransformerConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.embeddings = SpTransformerEmbedding(config)
        self.feature_encoders = nn.ModuleList(
            [SpTransformerFeatureEncoder(c["hidden_size"], config) for c in config.encoders]
        )
        self.projection = SpTransformerProjection(config)
        self.encoder = SpTransformerEncoder(config)
        self.prediction = SpTransformerPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_splice_labels == 3:
            splice_channels = ["no_splice", "acceptor", "donor"]
        else:
            splice_channels = [f"splice_label_{index}" for index in range(self.config.num_splice_labels)]
        return splice_channels + list(self.config.tissue_names)

    def postprocess(self, outputs: SpTransformerModelOutput | ModelOutput | Tensor) -> tuple[Tensor, list[str]]:
        r"""
        Return SpTransformer splice-site probabilities and tissue-usage scores with semantic channel names.

        Args:
            outputs: The output of [`SpTransformerModel`][multimolecule.models.SpTransformerModel], or its `logits`
                tensor.

        Returns:
            A tuple of `(scores, channels)`. The splice-site channels are softmax-normalized; tissue-usage channels
            are returned in the model's native scale.
        """
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]
        splice = logits[..., : self.config.num_splice_labels].softmax(dim=-1)
        usage = logits[..., self.config.num_splice_labels : self.config.num_splice_labels + self.config.num_tissues]
        return torch.cat([splice, usage], dim=-1), self.output_channels

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SpTransformerModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        output_contexts = kwargs.get("output_contexts", self.config.output_contexts)
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)

        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
        if isinstance(inputs_embeds, NestedTensor):
            if attention_mask is None:
                attention_mask = inputs_embeds.mask
            inputs_embeds = inputs_embeds.tensor

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        target_output_len = embedding_output.size(2) - 2 * self.config.context
        max_seq_len = self.config.max_seq_len
        odd_fix = embedding_output.size(2) & 1

        projected = self.projection(embedding_output)
        projected = _pad_or_crop_features(projected, max_seq_len, odd_fix)

        features = [feature_encoder(embedding_output) for feature_encoder in self.feature_encoders]
        if features:
            features = torch.cat(features, dim=1)
            features = _pad_or_crop_features(features, max_seq_len, odd_fix)
            hidden_state = torch.cat([features, projected], dim=1)
        else:
            hidden_state = projected
        hidden_state = self.projection.fuse(hidden_state)
        encoder_outputs = self.encoder(
            hidden_state,
            output_attentions=bool(output_attentions),
            output_hidden_states=bool(output_hidden_states),
        )
        context = encoder_outputs.last_hidden_state

        logits = self.prediction(context)

        logits = _pad_or_crop_outputs(logits, target_output_len, odd_fix)
        context = context.transpose(1, 2)
        context = _pad_or_crop_outputs(context, target_output_len, odd_fix)
        hidden_states = None
        if output_hidden_states and encoder_outputs.hidden_states is not None:
            hidden_states = tuple(
                _pad_or_crop_outputs(hidden_state, target_output_len, odd_fix)
                for hidden_state in encoder_outputs.hidden_states
            )

        return SpTransformerModelOutput(
            last_hidden_state=context,
            logits=logits,
            contexts=(context,) if output_contexts else None,
            hidden_states=hidden_states,
        )


class SpTransformerForTokenPrediction(SpTransformerPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import (
        ...     SpTransformerConfig,
        ...     SpTransformerFeatureEncoderConfig,
        ...     SpTransformerForTokenPrediction,
        ... )
        >>> encoder = SpTransformerFeatureEncoderConfig(hidden_size=4)
        >>> config = SpTransformerConfig(
        ...     context=2, hidden_size=8, encoders=[encoder], attention_hidden_size=16,
        ...     num_hidden_layers=1, num_attention_heads=2, num_local_attention_heads=1,
        ...     intermediate_size=32, bucket_size=4, max_seq_len=8, num_tissues=2, num_labels=2,
        ... )
        >>> model = SpTransformerForTokenPrediction(config)
        >>> input_ids = torch.randint(5, (1, 8))
        >>> output = model(input_ids, labels=torch.rand(1, 8, 2))
        >>> output["logits"].shape
        torch.Size([1, 8, 2])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MseLossBackward0>)
    """

    def __init__(self, config: SpTransformerConfig):
        super().__init__(config)
        self.model = SpTransformerModel(config)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SpTransformerTokenPredictorOutput:
        head_attention_mask = attention_mask
        if input_ids is None and inputs_embeds is not None and head_attention_mask is None:
            if isinstance(inputs_embeds, NestedTensor):
                head_attention_mask = inputs_embeds.mask
            else:
                head_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.int, device=inputs_embeds.device)

        outputs = self.model(
            input_ids,
            attention_mask=head_attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output = self.token_head(outputs, head_attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return SpTransformerTokenPredictorOutput(
            loss=loss,
            logits=logits,
            contexts=outputs.contexts,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SpTransformerEmbedding(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.context = config.context
        self.padding = config.context
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            inputs_embeds = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        batch_size = inputs_embeds.size(0)
        padding = torch.zeros(
            batch_size, self.vocab_size, self.padding, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs_embeds = torch.cat([padding, inputs_embeds, padding], dim=2)
        return inputs_embeds


class SpTransformerFeatureEncoder(nn.Module):
    """A SpliceAI-style dilated-residual convolutional feature extractor.

    SpTransformer reuses two such pre-trained feature encoders. Only the feature map (the skip aggregation, before
    the encoder's own output projections) is consumed by the attention block.
    """

    kernel_sizes = (11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21)
    dilations = (1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20)

    def __init__(self, hidden_size: int, config: SpTransformerConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(config.vocab_size, hidden_size, 1)
        self.skip = nn.Conv1d(hidden_size, hidden_size, 1)
        self.resblocks = nn.ModuleList(
            [
                SpTransformerResidualBlock(hidden_size, kernel_size, dilation, config)
                for kernel_size, dilation in zip(self.kernel_sizes, self.dilations)
            ]
        )
        self.convs = nn.ModuleList()
        for i in range(len(self.kernel_sizes)):
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.kernel_sizes)):
                self.convs.append(nn.Conv1d(hidden_size, hidden_size, 1))

    def forward(self, hidden_state: Tensor) -> Tensor:
        conv = self.conv1(hidden_state)
        skip = self.skip(conv)
        j = 0
        for i, residual_block in enumerate(self.resblocks):
            conv = residual_block(conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == len(self.kernel_sizes)):
                skip = skip + self.convs[j](conv)
                j += 1
        return skip


class SpTransformerResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilation: int, config: SpTransformerConfig):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        if isinstance(config.hidden_act, str):
            act1 = ACT2FN[config.hidden_act]
            act2 = ACT2FN[config.hidden_act]
        else:
            act1 = config.hidden_act
            act2 = config.hidden_act
        self.norm1 = nn.BatchNorm1d(hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act1 = act1
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding=padding)
        self.norm2 = nn.BatchNorm1d(hidden_size, config.batch_norm_eps, config.batch_norm_momentum)
        self.act2 = act2
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding=padding)

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class SpTransformerProjection(nn.Module):
    """Trainable input path and feature-fusion projection."""

    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(config.vocab_size, config.hidden_size, 1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, 1)
        encoder_size = sum(c["hidden_size"] for c in config.encoders)
        self.conv = nn.Conv1d(config.hidden_size + encoder_size, config.attention_hidden_size, 1)

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_state = self.conv1(inputs_embeds)
        hidden_state = self.conv2(hidden_state)
        return hidden_state

    def fuse(self, hidden_state: Tensor) -> Tensor:
        return self.conv(hidden_state)


class SpTransformerEncoder(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.position_embeddings = SpTransformerAxialPositionEmbeddings(config)
        self.layer = nn.ModuleList([SpTransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> SpTransformerEncoderOutput:
        hidden_state = hidden_state.transpose(1, 2).contiguous()
        hidden_state = hidden_state + self.position_embeddings(hidden_state)
        all_hidden_states: tuple[Tensor, ...] | None = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)  # type: ignore[operator]
        for layer in self.layer:
            if self.gradient_checkpointing and self.training and not output_attentions:
                hidden_state = self._gradient_checkpointing_func(layer.__call__, hidden_state)
            else:
                hidden_state = layer(hidden_state, output_attentions=output_attentions)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)  # type: ignore[operator]
        hidden_state = self.layer_norm(hidden_state)
        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-1] + (hidden_state,)  # type: ignore[index]
        last_hidden_state = hidden_state.transpose(1, 2).contiguous()
        return SpTransformerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states)


class SpTransformerAxialPositionEmbeddings(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        rows = config.max_seq_len // config.bucket_size
        cols = config.bucket_size
        self.shape = (rows, cols)
        self.max_seq_len = rows * cols
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, rows, 1, dim)),
                nn.Parameter(torch.zeros(1, 1, cols, dim)),
            ]
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        batch, seq_len, _ = hidden_state.shape
        embeddings = []
        for weight in self.weights:
            dim = weight.shape[-1]
            emb = weight.expand(batch, *self.shape, dim).reshape(batch, self.max_seq_len, dim)
            embeddings.append(emb)
        pos_emb = embeddings[0] + embeddings[1]
        return pos_emb[:, :seq_len].to(hidden_state)


class SpTransformerLayer(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        self.attention = SpTransformerAttention(config)
        self.intermediate = SpTransformerIntermediate(config)
        self.output = SpTransformerOutput(config)

    def forward(self, hidden_state: Tensor, output_attentions: bool = False) -> Tensor:
        attention_output, _ = self.attention(hidden_state, output_attentions=output_attentions)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SpTransformerAttention(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.layer_norm = nn.LayerNorm(dim)
        self.self = SpTransformerSelfAttention(config)
        self.output = SpTransformerSelfOutput(config)

    def forward(
        self, hidden_state: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, SpTransformerAttentionMap | None]:
        self_outputs = self.self(self.layer_norm(hidden_state), output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], hidden_state)
        return attention_output, self_outputs[1]


class SpTransformerSelfAttention(nn.Module):
    """Self attention with a mix of local (windowed) heads and Sinkhorn (sorted-bucket) heads.

    Reproduces the inference-time behavior of the upstream `sinkhorn-transformer` block as configured by
    SpTransformer: ``non_permutative=True`` with the parameter-free attention sort net, ``n_sortcut=0`` and
    one top bucket. The upstream code reuses the reordered keys as the reordered values when ``n_sortcut==0``;
    this checkpoint-faithful quirk is preserved here.
    """

    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = config.num_local_attention_heads
        self.num_sinkhorn_heads = self.num_heads - self.num_local_heads
        self.head_dim = dim // self.num_heads
        self.bucket_size = config.bucket_size
        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, dim * 2, bias=False)

    def forward(
        self, hidden_state: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, SpTransformerAttentionMap | None]:
        batch, seq_len, _ = hidden_state.shape
        heads, head_dim = self.num_heads, self.head_dim

        query = self._split_heads(self.query(hidden_state))
        key, value = (self._split_heads(t) for t in self.key_value(hidden_state).chunk(2, dim=-1))

        local_query, sinkhorn_query = query[:, : self.num_local_heads], query[:, self.num_local_heads :]
        local_key, sinkhorn_key = key[:, : self.num_local_heads], key[:, self.num_local_heads :]
        local_value, sinkhorn_value = value[:, : self.num_local_heads], value[:, self.num_local_heads :]

        record = output_attentions
        local_attentions = local_key_positions = None
        sinkhorn_attentions = sinkhorn_reorder = None

        outputs = []
        if self.num_local_heads > 0:
            local_out, local_attentions, local_key_positions = self._local_attention(
                local_query, local_key, local_value, record=record
            )
            outputs.append(local_out)
        if self.num_sinkhorn_heads > 0:
            sinkhorn_out, sinkhorn_attentions, sinkhorn_reorder = self._sinkhorn_attention(
                sinkhorn_query, sinkhorn_key, sinkhorn_value, record=record
            )
            outputs.append(sinkhorn_out)
        out = torch.cat(outputs, dim=1)

        out = out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)

        attention_map = None
        if record:
            attention_map = SpTransformerAttentionMap(
                local_attentions=local_attentions,
                local_key_positions=local_key_positions,
                sinkhorn_attentions=sinkhorn_attentions,
                sinkhorn_reorder=sinkhorn_reorder,
                bucket_size=self.bucket_size,
                look_backward=1,
                look_forward=1,
                num_local_heads=self.num_local_heads,
                num_sinkhorn_heads=self.num_sinkhorn_heads,
                sequence_length=seq_len,
            )
        return out, attention_map

    def _split_heads(self, hidden_state: Tensor) -> Tensor:
        batch, seq_len, _ = hidden_state.shape
        return hidden_state.view(batch, seq_len, self.num_heads, -1).transpose(1, 2)

    def _local_attention(
        self, query: Tensor, key: Tensor, value: Tensor, record: bool = False
    ) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        batch, heads, seq_len, head_dim = query.shape
        window = self.bucket_size
        windows = seq_len // window
        merged = batch * heads
        scale = head_dim**-0.5

        q = query.reshape(merged, windows, window, head_dim) * scale
        k = key.reshape(merged, windows, window, head_dim)
        v = value.reshape(merged, windows, window, head_dim)

        # Each window attends to itself plus the immediately preceding and following window.
        bk = _look_around(k)
        bv = _look_around(v)
        # `look_forward`/`look_backward` neighbouring windows are padded with -1; those padded
        # key positions are masked out (no rotary or exact-window masking is configured upstream).
        positions = torch.arange(seq_len, device=query.device).reshape(1, windows, window)
        bq_k = _look_around(positions.unsqueeze(-1)).reshape(1, windows, 1, -1)

        sim = torch.einsum("bwie,bwje->bwij", q, bk)
        mask_value = -torch.finfo(sim.dtype).max
        pad_mask = bq_k == -1
        sim = sim.masked_fill(pad_mask, mask_value)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bwij,bwje->bwie", attn, bv)
        out = out.reshape(batch, heads, seq_len, head_dim)

        attentions = key_positions = None
        if record:
            # `attn` is the exact per-window softmax used above; padded look-around columns already
            # carry weight 0. Reshape (B*H, windows, window, look*window) -> (B, H, windows, window, ...).
            look = bk.shape[2]
            attentions = attn.reshape(batch, heads, windows, window, look).detach()
            # Absolute source position for every key-axis column; padded columns are -1 (weight 0).
            key_positions = bq_k.reshape(windows, look).detach()
        return out, attentions, key_positions

    def _sinkhorn_attention(
        self, query: Tensor, key: Tensor, value: Tensor, record: bool = False
    ) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        batch, heads, seq_len, head_dim = query.shape
        bucket_size = self.bucket_size
        buckets = seq_len // bucket_size
        merged = batch * heads
        scale = head_dim**-0.5

        q = query.reshape(merged, seq_len, head_dim)
        k = key.reshape(merged, seq_len, head_dim)
        v = value.reshape(merged, seq_len, head_dim)

        b_q = q.reshape(merged, buckets, bucket_size, head_dim)
        b_k = k.reshape(merged, buckets, bucket_size, head_dim)
        b_v = v.reshape(merged, buckets, bucket_size, head_dim)

        sq = b_q.mean(dim=2)
        sk = b_k.mean(dim=2)
        reorder_scores = torch.einsum("bie,bje->bij", sq, sk) * (head_dim**-0.5)
        # differentiable_topk(R, k=1): softmax over kv buckets, keep only the argmax bucket weighted by its prob.
        scores = reorder_scores.softmax(dim=-1)
        values, indices = scores.topk(1, dim=-1)
        reorder = torch.zeros_like(scores).scatter_(-1, indices, values)

        b_k_r = torch.einsum("buv,bvtd->butd", reorder, b_k)
        # Upstream reuses the reordered keys as reordered values when n_sortcut == 0; preserved for parity.
        b_v_r = b_k_r

        keys = torch.cat([b_k_r, b_k], dim=2)
        values_cat = torch.cat([b_v_r, b_v], dim=2)

        dots = torch.einsum("buie,buje->buij", b_q, keys) * scale
        dots = dots.softmax(dim=-1)
        out = torch.einsum("buij,buje->buie", dots, values_cat)
        out = out.reshape(merged, seq_len, head_dim)
        out = out.reshape(batch, heads, seq_len, head_dim)

        attentions = reorder_map = None
        if record:
            # `dots` is the exact per-bucket softmax over the 2*bucket_size key axis
            # [reordered-bucket | own-bucket]. `reorder` is the exact bucket-permutation matrix
            # gathering the reordered keys, so columns 0:bucket_size map to source bucket
            # argmax(reorder[u]) and columns bucket_size:2*bucket_size map to bucket u itself.
            attentions = dots.reshape(batch, heads, buckets, bucket_size, 2 * bucket_size).detach()
            reorder_map = reorder.reshape(batch, heads, buckets, buckets).detach()
        return out, attentions, reorder_map


class SpTransformerSelfOutput(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.dense = nn.Linear(dim, dim)

    def forward(self, hidden_state: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        return hidden_state + input_tensor


class SpTransformerIntermediate(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.layer_norm = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.intermediate_act] if isinstance(config.intermediate_act, str) else config.intermediate_act
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.layer_norm(hidden_state)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.intermediate_act_fn(hidden_state)
        return hidden_state


class SpTransformerOutput(nn.Module):
    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.dense = nn.Linear(config.intermediate_size, dim)

    def forward(self, hidden_state: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        return hidden_state + input_tensor


class SpTransformerPredictionHead(nn.Module):
    """Original SpTransformer output heads.

    The network predicts, for each position, a 3-channel splice-site score (no-splice / acceptor / donor)
    and a per-tissue splice-site usage score. The concatenation reproduces the upstream output channels.
    The downstream task head is the shared [`TokenPredictionHead`].
    """

    def __init__(self, config: SpTransformerConfig):
        super().__init__()
        dim = config.attention_hidden_size
        self.splice = nn.Conv1d(dim, config.num_splice_labels, 1)
        self.usage = nn.Conv1d(dim, config.num_tissues, 1)

    def forward(self, context: Tensor) -> Tensor:
        splice = self.splice(context)
        usage = self.usage(context)
        return torch.cat([splice, usage], dim=1).transpose(1, 2)


def _pad_or_crop_features(hidden_state: Tensor, target_length: int, odd_fix: int) -> Tensor:
    seq_len = hidden_state.size(-1)
    pad = (-(seq_len - target_length)) // 2
    return F.pad(hidden_state, (pad, pad + odd_fix))


def _pad_or_crop_outputs(hidden_state: Tensor, target_length: int, odd_fix: int) -> Tensor:
    seq_len = hidden_state.size(1)
    pad = (-(seq_len - target_length)) // 2
    return F.pad(hidden_state.transpose(1, 2), (pad + odd_fix, pad)).transpose(1, 2)


def _look_around(x: Tensor, backward: int = 1, forward: int = 1, pad_value: int = -1) -> Tensor:
    """For a `(batch, windows, window, dim)` tensor, concatenate each window with its `backward`/`forward`
    neighbouring windows along the token axis, padding boundaries with `pad_value`.

    This reproduces the inference-time behaviour of `local_attention.look_around` as used by SpTransformer.
    """
    dim = 2
    pad = (len(x.shape) - dim) * (0, 0)
    padded = F.pad(x, (*pad, backward, forward), value=pad_value)
    tensors = padded.unfold(1, forward + backward + 1, 1)
    return tensors.movedim(-1, dim).flatten(dim, dim + 1)


@dataclass
class SpTransformerAttentionMap(ModelOutput):
    r"""
    Faithful, structured attention weights for **one** SpTransformer attention layer.

    SpTransformer's attention layer (`SpTransformerSelfAttention`) is *not* dense self-attention. It
    splits the heads into two groups with fundamentally different sparse-attention structures, so there is **no
    single dense `(batch, heads, seq, seq)` tensor** that faithfully represents the computation. Fabricating one
    (e.g. by scattering the block weights into a zero-filled `seq x seq` grid) would be a misleading
    interpretability artifact. Instead, this object exposes the *actual* `softmax` weights computed in the
    forward pass for each attention type, plus the indexing/permutation needed to map them back to absolute
    sequence positions.

    Conventions: ``B`` = batch, ``S`` = sequence length, ``W`` = ``window_size`` = ``bucket_size``,
    ``num_windows`` = ``S // W``, ``num_buckets`` = ``S // W``. Local heads come first along the head axis,
    Sinkhorn heads second, matching the split inside `SpTransformerSelfAttention`.

    Args:
        local_attentions (`torch.FloatTensor` of shape
            `(B, num_local_heads, num_windows, W, (look_backward + 1 + look_forward) * W)`, *optional*):
            Per-window softmax weights of the *windowed-local* heads. For window ``w``, query position ``i``
            (a token at absolute position ``w * W + i``) attends to a key axis that is the concatenation of
            windows ``[w - look_backward, ..., w, ..., w + look_forward]`` (each of length ``W``). Out-of-range
            neighbour windows are zero-padded and *masked* (their softmax weight is exactly ``0``). Use
            `local_key_positions` to recover the absolute source position of every key-axis column. `None` when
            the layer has no local heads.
        local_key_positions (`torch.LongTensor` of shape
            `(num_windows, (look_backward + 1 + look_forward) * W)`, *optional*):
            Absolute source sequence position for each key-axis column of `local_attentions`. Padded
            (out-of-range) columns are marked with ``-1`` and always carry softmax weight ``0``. Shared across
            batch and heads.
        sinkhorn_attentions (`torch.FloatTensor` of shape
            `(B, num_sinkhorn_heads, num_buckets, W, 2 * W)`, *optional*):
            Per-bucket softmax weights of the *Sinkhorn sorted-bucket* heads. For query bucket ``u``, the key
            axis (length ``2 * W``) is the concatenation of (a) the *single sorted/reordered* key bucket
            selected for ``u`` (columns ``0 : W``) and (b) ``u``'s own local bucket (columns ``W : 2 * W``).
            Map columns back to sequence positions via `sinkhorn_reorder` (for the first half) and
            ``u * W + j`` (for the second half). `None` when the layer has no Sinkhorn heads.
        sinkhorn_reorder (`torch.FloatTensor` of shape `(B, num_sinkhorn_heads, num_buckets, num_buckets)`,
            *optional*):
            The bucket-permutation / sort matrix produced by the parameter-free attention-sort net
            (``differentiable_topk(R, k=1)``). Row ``u`` is a one-hot-like (weighted) row whose nonzero entry
            ``v`` means query bucket ``u``'s reordered key bucket (columns ``0 : W`` of `sinkhorn_attentions`)
            is source bucket ``v`` (absolute positions ``v * W : v * W + W``), scaled by that entry. This is
            exactly the matrix used to gather the reordered keys in the forward pass. `None` when the layer has
            no Sinkhorn heads.
        bucket_size (`int`): ``W``, the local-attention window size and Sinkhorn bucket size.
        look_backward (`int`): number of preceding windows each local window attends to (``1`` upstream).
        look_forward (`int`): number of following windows each local window attends to (``1`` upstream).
        num_local_heads (`int`): number of windowed-local heads (first heads along the head axis).
        num_sinkhorn_heads (`int`): number of Sinkhorn sorted-bucket heads (remaining heads).
        sequence_length (`int`): ``S``, the attention-block sequence length these weights were computed on.

    Faithfulness guarantee: re-deriving the per-type attention output by contracting these exact softmax
    weights with the (block-gathered) values reproduces the layer's attention output bit-for-bit.
    """

    local_attentions: torch.FloatTensor | None = None
    local_key_positions: torch.LongTensor | None = None
    sinkhorn_attentions: torch.FloatTensor | None = None
    sinkhorn_reorder: torch.FloatTensor | None = None
    bucket_size: int | None = None
    look_backward: int | None = None
    look_forward: int | None = None
    num_local_heads: int | None = None
    num_sinkhorn_heads: int | None = None
    sequence_length: int | None = None


@dataclass
class SpTransformerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SpTransformerModelOutput(ModelOutput):
    """
    Base class for outputs of the SpTransformer model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, attention_hidden_size)`):
            Per-position attention-block representation. Consumed by [`TokenPredictionHead`].
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_splice_labels + num_tissues)`):
            Original SpTransformer per-position splice-site score (no-splice / acceptor / donor) and per-tissue
            splice-site usage score outputs.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True` is passed or when
            `config.output_contexts=True`):
            Tuple with the per-position attention-block representation of shape `(batch_size, sequence_length,
            attention_hidden_size)`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Attention-block hidden states before the first layer and after each layer, cropped or padded to the
            predicted sequence length. The final entry is the same representation as `last_hidden_state`.
        attentions (`tuple(SpTransformerAttentionMap)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            One [`SpTransformerAttentionMap`] per attention layer (in forward order). SpTransformer mixes
            *windowed-local* heads and *Sinkhorn sorted-bucket* heads, which are heterogeneous sparse attention
            patterns that **cannot** be faithfully flattened into a single dense `(batch, heads, seq, seq)` tensor.
            Each [`SpTransformerAttentionMap`] therefore exposes the *real, structured* softmax weights actually
            used in the forward pass, together with the index/permutation metadata needed to map them back to
            sequence positions. See [`SpTransformerAttentionMap`] for the exact schema. **These are not dense
            attention matrices**; treating them as such would misrepresent the computation.
    """

    last_hidden_state: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[SpTransformerAttentionMap, ...] | None = None


@dataclass
class SpTransformerTokenPredictorOutput(ModelOutput):
    """
    Base class for outputs of SpTransformer token prediction models.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Token prediction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Per-nucleotide prediction outputs.
        contexts (`tuple(torch.FloatTensor)`, *optional*, returned when `output_contexts=True`):
            Per-position attention-block representations.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Attention-block hidden states before the first layer and after each layer.
        attentions (`tuple(SpTransformerAttentionMap)`, *optional*, returned when `output_attentions=True`):
            Structured sparse-attention weights for each attention layer.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contexts: tuple[torch.FloatTensor, ...] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[SpTransformerAttentionMap, ...] | None = None


# Expose per-layer *faithful structured* attention weights for interpretability via the Transformers
# output-capture mechanism. Unlike a dense self-attention model, SpTransformer's attention layer mixes
# windowed-local and Sinkhorn sorted-bucket sparse patterns, so each layer surfaces a
# `SpTransformerAttentionMap` (not a dense `(B, H, S, S)` tensor, which would misrepresent the
# computation). `SpTransformerAttention.forward` returns `(output, attention_map)`, so the recorder captures
# the map at tuple index 1. Recording is opt-in (`output_attentions=True` /
# `config.output_attentions`); when inactive no map is built and the default forward computation and
# numerics are byte-for-byte unchanged.
SpTransformerPreTrainedModel._can_record_outputs = {
    "attentions": OutputRecorder(SpTransformerAttention, index=1, layer_name="attention"),
}
