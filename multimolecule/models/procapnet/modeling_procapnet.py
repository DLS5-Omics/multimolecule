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
from typing import Any

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

from multimolecule.modules import TokenPredictionHead

from ..modeling_outputs import TokenPredictorOutput
from .configuration_procapnet import ProCapNetConfig


class ProCapNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProCapNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ProCapNetLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)


class ProCapNetModel(ProCapNetPreTrainedModel):
    """
    The bare ProCapNet dilated-convolution backbone producing per-position features.

    Examples:
        >>> from multimolecule import ProCapNetConfig, ProCapNetModel, DnaTokenizer
        >>> config = ProCapNetConfig()
        >>> model = ProCapNetModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/procapnet")
        >>> input = tokenizer(("ACGT" * 529)[:2114], return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 2114, 512])
    """

    def __init__(self, config: ProCapNetConfig):
        super().__init__(config)
        self.embeddings = ProCapNetEmbedding(config)
        self.encoder = ProCapNetEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ProCapNetModelOutput:
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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(embedding_output, **kwargs)
        last_hidden_state = encoder_output.last_hidden_state.transpose(1, 2)

        return ProCapNetModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_output.hidden_states,
        )


class ProCapNetForProfilePrediction(ProCapNetPreTrainedModel):
    """
    ProCapNet with the factorized profile/count head for base-resolution PRO-cap signal prediction.

    This is a token/positional-prediction model: it is registered with the token AutoModel family and predicts a
    per-position value for every input nucleotide. The single base-resolution PRO-cap transcription-initiation task is
    factorized into two terminal branches sharing the backbone:

    - `profile_logits`: per-position, two-stranded multinomial logits of shape
      `(batch_size, profile_length, num_strands)`;
    - `count_logits`: a single strand-merged log-count scalar of shape `(batch_size, 1)`.

    Unlike single-stranded BPNet, the ProCapNet profile is a joint multinomial over **both strands and all
    positions** (the plus / minus strand share one count), so [`postprocess`][multimolecule.models.
    ProCapNetForProfilePrediction.postprocess] normalizes the profile over the strand-and-position axes jointly.

    Examples:
        >>> import torch
        >>> from multimolecule import ProCapNetConfig, ProCapNetForProfilePrediction, DnaTokenizer
        >>> config = ProCapNetConfig()
        >>> model = ProCapNetForProfilePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/procapnet")
        >>> input = tokenizer(("ACGT" * 529)[:2114], return_tensors="pt")
        >>> output = model(**input)
        >>> output["profile_logits"].shape
        torch.Size([1, 1000, 2])
        >>> output["count_logits"].shape
        torch.Size([1, 1])
        >>> track = model.postprocess(output)
        >>> track.shape
        torch.Size([1, 1000, 2])
    """

    def __init__(self, config: ProCapNetConfig):
        super().__init__(config)
        self.model = ProCapNetModel(config)
        self.profile_count_head = ProCapNetProfileCountHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: dict[str, Tensor] | tuple[Tensor, Tensor] | None = None,
        # labels: dict {"profile": (batch, profile_len, num_strands) int counts (raw),
        #               "count": (batch, 1) raw total counts (internally log1p-transformed)}
        #         or a (profile, count) tuple in the same order.
        **kwargs: Unpack[TransformersKwargs],
    ) -> ProCapNetProfilePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        head_output = self.profile_count_head(outputs.last_hidden_state, labels)

        return ProCapNetProfilePredictorOutput(
            loss=head_output.loss,
            profile_logits=head_output.profile_logits,
            count_logits=head_output.count_logits,
            hidden_states=outputs.hidden_states,
        )

    def postprocess(self, outputs: ProCapNetProfilePredictorOutput | ModelOutput) -> Tensor:
        r"""
        Recombine the factorized profile and count branches into the usable base-resolution track.

        ProCapNet does not predict the signal track directly; the profile branch predicts the *shape* and the count
        branch predicts the *total magnitude* (in log space). Because ProCapNet is two-stranded with a single
        strand-merged count, the profile is a joint multinomial over **both strands and all positions**. The usable
        prediction recombines them as `softmax(profile_logits, strands & positions) * expm1(count_logits)`, where
        `expm1` inverts the `log1p` count target the model is trained against.

        Args:
            outputs: The output of
                [`ProCapNetForProfilePrediction`][multimolecule.models.ProCapNetForProfilePrediction].

        Returns:
            The predicted base-resolution track of shape `(batch_size, profile_length, num_strands)`.
        """
        profile_logits = outputs["profile_logits"]
        count_logits = outputs["count_logits"]
        batch_size = profile_logits.shape[0]
        profile = F.softmax(profile_logits.reshape(batch_size, -1), dim=-1).reshape(profile_logits.shape)
        return profile * torch.expm1(count_logits).unsqueeze(1)


class ProCapNetForTokenPrediction(ProCapNetPreTrainedModel):
    """
    ProCapNet backbone with a randomly initialized generic token-prediction head.

    This class is intended for downstream fine-tuning from the ProCapNet backbone. It returns the standard
    [`TokenPredictorOutput`][multimolecule.models.TokenPredictorOutput] with a single `logits` field, unlike
    [`ProCapNetForProfilePrediction`][multimolecule.models.ProCapNetForProfilePrediction], which exposes the
    published factorized `profile_logits` / `count_logits` task head.

    Examples:
        >>> import torch
        >>> from multimolecule import ProCapNetConfig, ProCapNetForTokenPrediction
        >>> config = ProCapNetConfig()
        >>> model = ProCapNetForTokenPrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, 16))
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([1, 16, 2])
    """

    def __init__(self, config: ProCapNetConfig):
        super().__init__(config)
        self.model = ProCapNetModel(config)
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
    ) -> tuple[Tensor, ...] | TokenPredictorOutput:
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
        return TokenPredictorOutput(
            loss=output.loss,
            logits=output.logits,
            hidden_states=outputs.hidden_states,
        )


class ProCapNetEmbedding(nn.Module):
    """One-hot encode `input_ids` into `(batch_size, vocab_size, sequence_length)` channel-first features."""

    def __init__(self, config: ProCapNetConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
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
            if input_ids is None:
                raise ValueError("input_ids must be provided when inputs_embeds is None")
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            inputs_embeds = F.one_hot(
                input_ids.clamp(min=0, max=self.vocab_size - 1),
                num_classes=self.vocab_size,
            ).to(dtype=dtype)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype=dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class ProCapNetEncoder(nn.Module):
    def __init__(self, config: ProCapNetConfig):
        super().__init__()
        self.config = config
        self.stem = ProCapNetStem(config)
        self.layer = nn.ModuleList(
            [ProCapNetLayer(config, dilation=2 ** (i + 1)) for i in range(config.num_dilated_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ProCapNetEncoderOutput:
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        all_hidden_states: tuple[Tensor, ...] | None = () if output_hidden_states else None

        hidden_state = self.stem(hidden_state)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]
        for layer_module in self.layer:
            if self.gradient_checkpointing and self.training:
                hidden_state = self._gradient_checkpointing_func(layer_module.__call__, hidden_state)
            else:
                hidden_state = layer_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state.transpose(1, 2),)  # type: ignore[operator]

        return ProCapNetEncoderOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
        )


class ProCapNetStem(nn.Module):
    """First (motif) convolution mapping one-hot channels into the backbone feature space."""

    def __init__(self, config: ProCapNetConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.vocab_size,
            config.hidden_size,
            kernel_size=config.stem_kernel_size,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.act(self.conv(hidden_state))


class ProCapNetLayer(nn.Module):
    """Dilated residual convolution block: `out = in + act(conv(in))`."""

    def __init__(self, config: ProCapNetConfig, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.dilated_kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return hidden_state + self.act(self.conv(hidden_state))


class ProCapNetProfileCountHead(nn.Module):
    r"""
    The factorized ProCapNet head owning the two terminal branches over the shared backbone features.

    - The profile branch is a wide convolution producing per-position, two-stranded logits of shape
      `(batch_size, profile_length, num_strands)`. It is trained with a multinomial negative log-likelihood over the
      joint strand-and-position distribution (ProCapNet normalizes plus / minus strands together).
    - The count branch global-average-pools the center-cropped profile window (not the full backbone) and applies a
      linear layer producing a single strand-merged log-count scalar of shape `(batch_size, 1)`. It is trained with
      mean-squared error on the `log(count + 1)` total.

    The composite loss is `profile_nll + count_loss_weight * count_mse`.
    """

    def __init__(self, config: ProCapNetConfig):
        super().__init__()
        self.num_strands = config.num_strands
        self.count_loss_weight = config.count_loss_weight
        self.profile_length = config.profile_length
        self.profile_kernel_size = config.profile_kernel_size
        self.profile = nn.Conv1d(
            config.hidden_size,
            config.num_strands,
            kernel_size=config.profile_kernel_size,
            padding=0,
        )
        self.count = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_state: Tensor,
        labels: dict[str, Tensor] | tuple[Tensor, Tensor] | None = None,
    ) -> ProCapNetHeadOutput:
        features = hidden_state.transpose(1, 2)
        profile_features = _center_crop_for_profile(
            features,
            self.profile_length,
            self.profile_kernel_size,
        )
        profile_logits = self.profile(profile_features).transpose(1, 2)
        pooled = profile_features.mean(dim=2)
        count_logits = self.count(pooled)

        loss = None
        if labels is not None:
            profile_labels, count_labels = _unpack_labels(labels)
            profile_loss = _multinomial_nll(profile_logits, profile_labels)
            count_loss = F.mse_loss(count_logits, _log1p(count_labels).to(count_logits.dtype))
            loss = profile_loss + self.count_loss_weight * count_loss

        return ProCapNetHeadOutput(profile_logits=profile_logits, count_logits=count_logits, loss=loss)


def _unpack_labels(
    labels: dict[str, Tensor] | tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    if isinstance(labels, dict):
        return labels["profile"], labels["count"]
    return labels[0], labels[1]


def _center_crop_for_profile(hidden_state: Tensor, profile_length: int, profile_kernel_size: int) -> Tensor:
    target_length = profile_length + profile_kernel_size - 1
    sequence_length = hidden_state.size(-1)
    if sequence_length < target_length:
        raise ValueError(
            f"ProCapNet needs at least {target_length} positions before the profile head, got {sequence_length}."
        )
    left = (sequence_length - target_length) // 2
    return hidden_state[..., left : left + target_length]


def _log1p(counts: Tensor) -> Tensor:
    """`log(count + 1)` matching the upstream ProCapNet ``log1pMSELoss`` target transform."""
    return torch.log(counts.to(torch.get_default_dtype()) + 1)


def _multinomial_nll(logits: Tensor, observed: Tensor) -> Tensor:
    r"""
    Strand-merged multinomial negative log-likelihood, averaged over the batch.

    `logits` and `observed` are `(batch_size, num_positions, num_strands)`. Unlike single-stranded BPNet, the
    ProCapNet multinomial distribution is jointly over the `num_positions * num_strands` flattened categories for
    each example (the plus / minus strands share one count), so the log-softmax and the count sums are taken over the
    flattened strand-and-position axes.
    """
    batch_size = logits.shape[0]
    logits = logits.reshape(batch_size, -1)
    observed = observed.reshape(batch_size, -1)
    log_probs = F.log_softmax(logits, dim=-1)
    total = observed.sum(dim=-1)
    log_fact_total = torch.lgamma(total + 1)
    log_fact_counts = torch.lgamma(observed + 1).sum(dim=-1)
    log_likelihood = log_fact_total - log_fact_counts + (observed * log_probs).sum(dim=-1)
    return -log_likelihood.mean()


@dataclass
class ProCapNetModelOutput(ModelOutput):
    """
    Base class for outputs of the ProCapNet backbone.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Per-position backbone features.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the stem output plus one per dilated layer) of shape `(batch_size,
            sequence_length, hidden_size)`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ProCapNetEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ProCapNetHeadOutput(ModelOutput):
    """
    Output of the factorized ProCapNet profile/count head.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Composite multinomial-NLL + weighted count-MSE loss.
        profile_logits (`torch.FloatTensor` of shape `(batch_size, profile_length, num_strands)`):
            Per-position, two-stranded multinomial logits.
        count_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Strand-merged log-count scalar.
    """

    loss: torch.FloatTensor | None = None
    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None


@dataclass
class ProCapNetProfilePredictorOutput(ModelOutput):
    """
    Base class for outputs of
    [`ProCapNetForProfilePrediction`][multimolecule.models.ProCapNetForProfilePrediction].

    The standard single-`logits` predictor dataclasses cannot express ProCapNet's factorized output, so this
    model-local dataclass exposes the two terminal branches separately. Use
    [`postprocess`][multimolecule.models.ProCapNetForProfilePrediction.postprocess] to recombine them.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Composite multinomial-NLL (profile) + weighted count-MSE (count) loss.
        profile_logits (`torch.FloatTensor` of shape `(batch_size, profile_length, num_strands)`):
            Per-position, two-stranded multinomial logits.
        count_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Strand-merged log-count scalar.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of backbone hidden states of shape `(batch_size, sequence_length, hidden_size)`.
    """

    loss: torch.FloatTensor | None = None
    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
