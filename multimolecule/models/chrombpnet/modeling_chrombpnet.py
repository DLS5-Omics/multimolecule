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
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults

from multimolecule.modules import TokenPredictionHead

from ..modeling_outputs import TokenPredictorOutput
from .configuration_chrombpnet import ChromBPNetConfig


class ChromBPNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChromBPNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["ChromBPNetLayer"]

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


class ChromBPNetModel(ChromBPNetPreTrainedModel):
    """
    The bare ChromBPNet model: an enzyme-bias sub-model composed with a bias-corrected accessibility sub-model.

    ChromBPNet predicts base-resolution chromatin accessibility (ATAC-seq / DNase-seq) with explicit enzyme-bias
    correction. It internally owns two BPNet-style dilated-convolution sub-models and composes them so the model
    exposes a single clean factorized profile/count output:

    - the *bias* sub-model captures the Tn5/DNase enzyme cleavage bias on chromatin background;
    - the *accessibility* sub-model learns the bias-corrected accessibility signal.

    The two sub-models are composed internally: their per-position profile logits are added, and their count logits are
    combined in log/exp space via `logsumexp`. The sub-model split is an implementation detail, not a user-facing API.

    Examples:
        >>> from multimolecule import ChromBPNetConfig, ChromBPNetModel, DnaTokenizer
        >>> config = ChromBPNetConfig()
        >>> model = ChromBPNetModel(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/chrombpnet")
        >>> input = tokenizer(("ACGT" * 529)[:2114], return_tensors="pt")
        >>> output = model(**input)
        >>> output["profile_logits"].shape
        torch.Size([1, 1000, 1])
        >>> output["count_logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: ChromBPNetConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embeddings = ChromBPNetEmbedding(config)
        self.accessibility = ChromBPNetBranch(
            config, hidden_size=config.hidden_size, num_dilated_layers=config.num_dilated_layers
        )
        self.bias = ChromBPNetBranch(
            config, hidden_size=config.bias_hidden_size, num_dilated_layers=config.bias_num_dilated_layers
        )
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ChromBPNetModelOutput:
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
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)

        accessibility_output = self.accessibility(embedding_output, output_hidden_states=output_hidden_states)
        bias_output = self.bias(embedding_output, output_hidden_states=output_hidden_states)

        # ChromBPNet composition (kundajelab/chrombpnet `chrombpnet_with_bias_model`):
        # profile logits are added; counts are combined in log/exp space via logsumexp.
        profile_logits = accessibility_output.profile_logits + bias_output.profile_logits
        count_logits = torch.logsumexp(
            torch.stack([accessibility_output.count_logits, bias_output.count_logits], dim=-1), dim=-1
        )

        hidden_states = None
        if output_hidden_states:
            hidden_states = (accessibility_output.hidden_states or ()) + (bias_output.hidden_states or ())

        return ChromBPNetModelOutput(
            last_hidden_state=accessibility_output.last_hidden_state,
            profile_logits=profile_logits,
            count_logits=count_logits,
            hidden_states=hidden_states,
        )


class ChromBPNetForProfilePrediction(ChromBPNetPreTrainedModel):
    """
    ChromBPNet with the factorized profile/count head for base-resolution chromatin-accessibility prediction.

    This is a token/positional-prediction model: it is registered with the token AutoModel family and predicts a
    per-position value for every input nucleotide. The single base-resolution task is factorized into two terminal
    branches:

    - `profile_logits`: per-position multinomial logits of shape `(batch_size, profile_length, num_labels)`;
    - `count_logits`: a scalar per task and strand of shape `(batch_size, num_labels)`,

    where `num_labels = num_tasks * num_strands`. Use [`postprocess`][multimolecule.models.
    ChromBPNetForProfilePrediction.postprocess] to recombine them into the usable base-resolution track.

    The enzyme-bias correction (the internal bias + accessibility composition) is performed inside
    [`ChromBPNetModel`][multimolecule.models.ChromBPNetModel]; the factorized head here mirrors BPNet and operates on
    the already bias-corrected, composed profile and count logits.

    Examples:
        >>> import torch
        >>> from multimolecule import ChromBPNetConfig, ChromBPNetForProfilePrediction, DnaTokenizer
        >>> config = ChromBPNetConfig()
        >>> model = ChromBPNetForProfilePrediction(config)
        >>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/chrombpnet")
        >>> input = tokenizer(("ACGT" * 529)[:2114], return_tensors="pt")
        >>> output = model(**input)
        >>> output["profile_logits"].shape
        torch.Size([1, 1000, 1])
        >>> output["count_logits"].shape
        torch.Size([1, 1])
        >>> track = model.postprocess(output)
        >>> track.shape
        torch.Size([1, 1000, 1])
    """

    def __init__(self, config: ChromBPNetConfig):
        super().__init__(config)
        self.model = ChromBPNetModel(config)
        self.profile_count_head = ChromBPNetProfileCountHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_tasks == 1 and self.config.num_strands == 1:
            return ["signal"]
        tasks = [f"task_{index}" for index in range(self.config.num_tasks)]
        if self.config.num_strands == 2:
            strands = ["plus", "minus"]
        else:
            strands = [f"strand_{index}" for index in range(self.config.num_strands)]
        return [f"{task}_{strand}" for task in tasks for strand in strands]

    @merge_with_config_defaults
    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: dict[str, Tensor] | Tuple[Tensor, Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ChromBPNetProfilePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        head_output = self.profile_count_head(outputs.profile_logits, outputs.count_logits, labels)

        return ChromBPNetProfilePredictorOutput(
            loss=head_output.loss,
            profile_logits=head_output.profile_logits,
            count_logits=head_output.count_logits,
            hidden_states=outputs.hidden_states,
        )

    def postprocess(self, outputs: ChromBPNetProfilePredictorOutput | ModelOutput) -> Tensor:
        r"""
        Recombine the factorized profile and count branches into the usable base-resolution track.

        ChromBPNet does not predict the accessibility track directly; the profile branch predicts the *shape* (a
        per-position multinomial distribution) and the count branch predicts the *total magnitude* (in log space).
        The usable prediction recombines them as `softmax(profile_logits, positions) * expm1(count_logits)`.

        Args:
            outputs: The output of
                [`ChromBPNetForProfilePrediction`][multimolecule.models.ChromBPNetForProfilePrediction].

        Returns:
            The predicted base-resolution track of shape `(batch_size, profile_length, num_labels)`.
        """
        profile_logits = outputs["profile_logits"]
        count_logits = outputs["count_logits"]
        profile = F.softmax(profile_logits, dim=1)
        return profile * torch.expm1(count_logits).unsqueeze(1)


class ChromBPNetForTokenPrediction(ChromBPNetPreTrainedModel):
    """
    ChromBPNet accessibility backbone with a randomly initialized generic token-prediction head.

    This class attaches the shared MultiMolecule token head to the accessibility sub-model representation and returns a
    standard single-`logits` output for downstream fine-tuning. The published ChromBPNet profile/count task remains
    exposed through [`ChromBPNetForProfilePrediction`][multimolecule.models.ChromBPNetForProfilePrediction].

    Examples:
        >>> import torch
        >>> from multimolecule import ChromBPNetConfig, ChromBPNetForTokenPrediction
        >>> config = ChromBPNetConfig()
        >>> model = ChromBPNetForTokenPrediction(config)
        >>> input_ids = torch.randint(config.vocab_size, (1, config.sequence_length))
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([1, 2114, 1])
    """

    def __init__(self, config: ChromBPNetConfig):
        super().__init__(config)
        self.model = ChromBPNetModel(config)
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
    ) -> Tuple[Tensor, ...] | TokenPredictorOutput:
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


class ChromBPNetBranch(nn.Module):
    """
    A single BPNet-style dilated-convolution sub-model with its own factorized profile/count terminal branches.

    ChromBPNet owns two of these: a *bias* sub-model and an *accessibility* sub-model. Each consists of a motif
    stem convolution, a stack of dilated residual convolutions, and the factorized profile (wide convolution) and count
    (global-average-pool + linear) terminal branches.
    """

    def __init__(self, config: ChromBPNetConfig, hidden_size: int, num_dilated_layers: int):
        super().__init__()
        self.config = config
        self.num_tasks = config.num_tasks
        self.num_strands = config.num_strands
        self.profile_length = config.profile_length
        self.profile_kernel_size = config.profile_kernel_size
        self.valid_trim = _valid_trim(config, num_dilated_layers)
        self.stem = ChromBPNetStem(config, hidden_size=hidden_size)
        self.layer = nn.ModuleList(
            [ChromBPNetLayer(config, hidden_size=hidden_size, dilation=2 ** (i + 1)) for i in range(num_dilated_layers)]
        )
        self.profile = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_size,
                    config.num_strands,
                    kernel_size=config.profile_kernel_size,
                    padding=0,
                )
                for _ in range(config.num_tasks)
            ]
        )
        self.count = nn.ModuleList([nn.Linear(hidden_size, config.num_strands) for _ in range(config.num_tasks)])
        self.gradient_checkpointing = False

    def forward(self, hidden_state: Tensor, output_hidden_states: bool = False) -> ChromBPNetBranchOutput:
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

        profile_features = _center_crop_for_profile(hidden_state, self.profile_length, self.profile_kernel_size)
        profile_logits = torch.cat([branch(profile_features) for branch in self.profile], dim=1).transpose(1, 2)
        count_features = _center_crop_for_count(hidden_state, self.valid_trim)
        pooled = count_features.mean(dim=2)
        count_logits = torch.cat([branch(pooled) for branch in self.count], dim=1)

        return ChromBPNetBranchOutput(
            last_hidden_state=hidden_state.transpose(1, 2),
            profile_logits=profile_logits,
            count_logits=count_logits,
            hidden_states=all_hidden_states,
        )


class ChromBPNetEmbedding(nn.Module):
    """One-hot encode `input_ids` into `(batch_size, vocab_size, sequence_length)` channel-first features."""

    def __init__(self, config: ChromBPNetConfig):
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
                raise ValueError("You have to specify input_ids when inputs_embeds is not provided")
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class ChromBPNetStem(nn.Module):
    """First (motif) convolution mapping one-hot channels into the backbone feature space."""

    def __init__(self, config: ChromBPNetConfig, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            config.vocab_size,
            hidden_size,
            kernel_size=config.stem_kernel_size,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.act(self.conv(hidden_state))


class ChromBPNetLayer(nn.Module):
    """Dilated residual convolution block: `out = in + act(conv(in))`."""

    def __init__(self, config: ChromBPNetConfig, hidden_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.dilated_kernel_size,
            dilation=dilation,
            padding="same",
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return hidden_state + self.act(self.conv(hidden_state))


class ChromBPNetProfileCountHead(nn.Module):
    r"""
    The factorized ChromBPNet head computing the composite loss over the already bias-corrected, composed logits.

    [`ChromBPNetModel`][multimolecule.models.ChromBPNetModel] performs the bias + accessibility composition (profile
    logits added, counts combined via `logsumexp`) and produces the final factorized `profile_logits` /
    `count_logits`. This head mirrors BPNet's `BPNetProfileCountHead`: it passes the composed logits through unchanged
    and, when labels are provided, computes the composite loss.

    - The profile branch is trained with a multinomial negative log-likelihood over positions (the per-position
      distribution shape).
    - The count branch is trained with mean-squared error on the log total count.

    The composite loss is `profile_nll + count_loss_weight * count_mse`.
    """

    def __init__(self, config: ChromBPNetConfig):
        super().__init__()
        self.num_tasks = config.num_tasks
        self.num_strands = config.num_strands
        self.count_loss_weight = config.count_loss_weight

    def forward(
        self,
        profile_logits: Tensor,
        count_logits: Tensor,
        labels: dict[str, Tensor] | Tuple[Tensor, Tensor] | None = None,
    ) -> ChromBPNetHeadOutput:
        loss = None
        if labels is not None:
            profile_labels, count_labels = _unpack_labels(labels)
            profile_loss = _multinomial_nll(profile_logits, profile_labels)
            count_loss = F.mse_loss(count_logits, count_labels.to(count_logits.dtype))
            loss = profile_loss + self.count_loss_weight * count_loss

        return ChromBPNetHeadOutput(profile_logits=profile_logits, count_logits=count_logits, loss=loss)


def _unpack_labels(
    labels: dict[str, Tensor] | Tuple[Tensor, Tensor],
) -> Tuple[Tensor, Tensor]:
    if isinstance(labels, dict):
        return labels["profile"], labels["count"]
    return labels[0], labels[1]


def _center_crop_for_profile(hidden_state: Tensor, profile_length: int, profile_kernel_size: int) -> Tensor:
    target_length = profile_length + profile_kernel_size - 1
    sequence_length = hidden_state.size(-1)
    if sequence_length < target_length:
        raise ValueError(
            f"ChromBPNet needs at least {target_length} positions before the profile head, got {sequence_length}."
        )
    left = (sequence_length - target_length) // 2
    return hidden_state[..., left : left + target_length]


def _center_crop_for_count(hidden_state: Tensor, valid_trim: int) -> Tensor:
    target_length = hidden_state.size(-1) - valid_trim
    if target_length < 1:
        raise ValueError(
            f"ChromBPNet needs at least {valid_trim + 1} positions before the count head, got {hidden_state.size(-1)}."
        )
    left = (hidden_state.size(-1) - target_length) // 2
    return hidden_state[..., left : left + target_length]


def _valid_trim(config: ChromBPNetConfig, num_dilated_layers: int) -> int:
    return (config.stem_kernel_size - 1) + sum(
        (config.dilated_kernel_size - 1) * 2 ** (i + 1) for i in range(num_dilated_layers)
    )


def _multinomial_nll(logits: Tensor, observed: Tensor) -> Tensor:
    r"""
    Per-position multinomial negative log-likelihood, averaged over the batch and label channels.

    `logits` and `observed` are `(batch_size, num_positions, num_labels)`; the multinomial distribution is over the
    `num_positions` positions for each `(batch, label)` pair.
    """
    log_probs = F.log_softmax(logits, dim=1)
    total = observed.sum(dim=1)
    log_fact_total = torch.lgamma(total + 1)
    log_fact_counts = torch.lgamma(observed + 1).sum(dim=1)
    log_likelihood = log_fact_total - log_fact_counts + (observed * log_probs).sum(dim=1)
    return -log_likelihood.mean()


@dataclass
class ChromBPNetModelOutput(ModelOutput):
    """
    Base class for outputs of the ChromBPNet backbone.

    The ChromBPNet backbone performs the bias + accessibility composition and exposes both the accessibility branch
    representation for generic fine-tuning and the composed factorized profile / count logits.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Accessibility branch backbone representation.
        profile_logits (`torch.FloatTensor` of shape `(batch_size, profile_length, num_labels)`):
            Composed (bias-corrected) per-position multinomial logits.
        count_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Composed per task/strand log-count scalars.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` backbone hidden states (accessibility sub-model first, then bias sub-model)
            of shape `(batch_size, sequence_length, hidden_size)`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ChromBPNetBranchOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ChromBPNetHeadOutput(ModelOutput):
    """
    Output of the factorized ChromBPNet profile/count head.

    Args:
        profile_logits (`torch.FloatTensor` of shape `(batch_size, profile_length, num_labels)`):
            Per-position multinomial logits.
        count_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Per task/strand log-count scalars.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Composite multinomial-NLL + weighted count-MSE loss.
    """

    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None


@dataclass
class ChromBPNetProfilePredictorOutput(ModelOutput):
    """
    Base class for outputs of
    [`ChromBPNetForProfilePrediction`][multimolecule.models.ChromBPNetForProfilePrediction].

    The standard single-`logits` predictor dataclasses cannot express ChromBPNet's factorized output, so this
    model-local dataclass exposes the two terminal branches separately. Use
    [`postprocess`][multimolecule.models.ChromBPNetForProfilePrediction.postprocess] to recombine them.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Composite multinomial-NLL (profile) + weighted count-MSE (count) loss.
        profile_logits (`torch.FloatTensor` of shape `(batch_size, profile_length, num_labels)`):
            Per-position multinomial logits, where `num_labels = num_tasks * num_strands`.
        count_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Per task/strand log-count scalars.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of backbone hidden states of shape `(batch_size, sequence_length, hidden_size)`.
    """

    loss: torch.FloatTensor | None = None
    profile_logits: torch.FloatTensor | None = None
    count_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
