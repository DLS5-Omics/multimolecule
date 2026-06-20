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
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import SequencePredictionHead

from ..modeling_outputs import SequencePredictorOutput
from .configuration_rbpeclip import RbpEclipConfig


class RbpEclipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RbpEclipConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["RbpEclipEncoder"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_uniform_(module.weight, a=5**0.5)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / fan_in**0.5 if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class RbpEclipModel(RbpEclipPreTrainedModel):
    """
    The bare RBP-eCLIP model returning the pooled sequence representation.

    Examples:
        >>> from multimolecule import RbpEclipConfig, RbpEclipModel, RnaTokenizer
        >>> config = RbpEclipConfig()
        >>> model = RbpEclipModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rbpeclip")
        >>> input = tokenizer("ACGU" * 25 + "A", return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([1, 100])
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__(config)
        self.embeddings = RbpEclipEmbedding(config)
        self.encoder = RbpEclipEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        position_features: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RbpEclipModelOutput:
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
        pooled_output = self.encoder(embedding_output, position_features=position_features)
        return RbpEclipModelOutput(pooler_output=pooled_output)


class RbpEclipForSequencePrediction(RbpEclipPreTrainedModel):
    """
    RBP-eCLIP with a sequence-level binding-score prediction head.

    Each trained Hub checkpoint corresponds to a single RNA-binding protein; the head outputs a scalar
    binding-strength logit for the input RNA peak window.

    Examples:
        >>> import torch
        >>> from multimolecule import RbpEclipConfig, RbpEclipForSequencePrediction, RnaTokenizer
        >>> config = RbpEclipConfig()
        >>> model = RbpEclipForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rbpeclip")
        >>> input = tokenizer("ACGU" * 25 + "A", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1.0]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MseLossBackward0>)
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__(config)
        self.model = RbpEclipModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        if self.config.num_labels != 1:
            return [f"binding_score_{index}" for index in range(self.config.num_labels)]
        return ["binding_score"]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        position_features: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_features=position_features,
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

    def postprocess(self, outputs: SequencePredictorOutput | ModelOutput) -> Tensor:
        # The upstream eCLIP RBP head emits a sigmoid binding probability; expose it as the pipeline-level
        # postprocess so downstream consumers see calibrated probabilities.
        return torch.sigmoid(outputs["logits"])


class RbpEclipEmbedding(nn.Module):
    """One-hot input projection for the fixed-length RBP-eCLIP peak window.

    The upstream Kipoi checkpoints one-hot encode RNA as `["A", "C", "G", "U"]`; in MultiMolecule the
    input channels are derived from the streamline RNA alphabet (`ACGUN`) and the upstream-to-MM channel
    permutation is handled in :mod:`convert_checkpoint`.
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.sequence_length = config.sequence_length
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
            self._check_sequence_length(input_ids.size(-1))
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype)
            invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
            if invalid.any():
                inputs_embeds = inputs_embeds * (~invalid).unsqueeze(-1).to(dtype)
        else:
            self._check_sequence_length(inputs_embeds.size(1))
            inputs_embeds = inputs_embeds.to(dtype)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        # (batch, vocab_size, sequence_length) for the 1D convolutional stack.
        return inputs_embeds.transpose(1, 2)

    def _check_sequence_length(self, sequence_length: int):
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"RBP-eCLIP expects fixed-length {self.sequence_length} nt inputs, but got {sequence_length}. "
                "Pad or crop the sequence to match the configured sequence_length."
            )


class RbpEclipEncoder(nn.Module):
    """Sequence + position encoder for the RBP-eCLIP model.

    The sequence module is a two-layer 1D convolutional stack (motif-scoring conv + 1x1 mixing conv)
    followed by strided max-pooling and a flatten. The optional position module evaluates eight scalar
    genomic-landmark distances through a pre-computed B-spline basis and a per-feature GAM 1x1
    convolution; the resulting `num_position_features * num_position_filters` scalars are concatenated
    with the flattened sequence features before the dense head.
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__()
        self.sequence_module = RbpEclipSequenceModule(config)
        self.position_module = RbpEclipPositionModule(config)
        self.activation = ACT2FN[config.hidden_act]
        self.use_batchnorm = config.use_batchnorm
        self.dropout = nn.Dropout(config.hidden_dropout)
        hidden_in = self.sequence_module.output_features + self.position_module.output_features
        self.dense = nn.Linear(hidden_in, config.num_hidden)
        if config.use_batchnorm:
            # Upstream applies `kl.BatchNormalization()` over the feature axis of the flattened pooled
            # sequence features (before concatenating the position scalars) and again after the hidden
            # dense layer.
            self.pooled_batch_norm = nn.BatchNorm1d(
                self.sequence_module.output_features,
                eps=config.batch_norm_eps,
                momentum=config.batch_norm_momentum,
            )
            self.dense_batch_norm = nn.BatchNorm1d(
                config.num_hidden,
                eps=config.batch_norm_eps,
                momentum=config.batch_norm_momentum,
            )
        self.gradient_checkpointing = False

    def forward(self, hidden_state: Tensor, position_features: Tensor | None = None) -> Tensor:
        sequence_features = self.sequence_module(hidden_state)
        if self.use_batchnorm:
            sequence_features = self.pooled_batch_norm(sequence_features)
        sequence_features = self.dropout(sequence_features)
        position_summary = self.position_module(position_features, batch_size=sequence_features.size(0))
        if position_summary is not None:
            shared = torch.cat([sequence_features, position_summary], dim=-1)
        else:
            shared = sequence_features
        hidden = self.dense(shared)
        hidden = self.activation(hidden)
        if self.use_batchnorm:
            hidden = self.dense_batch_norm(hidden)
        hidden = self.dropout(hidden)
        return hidden


class RbpEclipSequenceModule(nn.Module):
    """Two-layer 1D convolutional stack + strided max-pool + flatten over the RNA peak window.

    The upstream model places `kl.BatchNormalization(axis=1)` after each convolution, which in Keras's
    `(batch, length, channels)` Conv1D output means batch-normalising the *length* axis (one set of
    affine parameters per output position). The PyTorch equivalent transposes to `(batch, length,
    channels)` and uses a `BatchNorm1d(num_features=post_conv1_length)`.
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__()
        self.filters = config.num_sequence_filters
        self.conv1 = nn.Conv1d(
            config.vocab_size,
            config.num_sequence_filters,
            kernel_size=config.sequence_kernel_size,
        )
        self.conv2 = nn.Conv1d(
            config.num_sequence_filters,
            config.num_sequence_filters,
            kernel_size=1,
        )
        self.activation = ACT2FN[config.hidden_act]
        self.use_batchnorm = config.use_batchnorm
        self.post_conv1_length = config.post_conv1_length
        if config.use_batchnorm:
            # Keras `axis=1` BN over the Conv1D output `(batch, length, channels)` keeps one affine
            # parameter per output position. The PyTorch equivalent is `BatchNorm1d(post_conv1_length)`
            # applied to a `(batch, post_conv1_length, channels)` tensor.
            self.batch_norm1 = nn.BatchNorm1d(
                config.post_conv1_length,
                eps=config.batch_norm_eps,
                momentum=config.batch_norm_momentum,
            )
            self.batch_norm2 = nn.BatchNorm1d(
                config.post_conv1_length,
                eps=config.batch_norm_eps,
                momentum=config.batch_norm_momentum,
            )
        self.pool = nn.MaxPool1d(config.sequence_pool_size)
        self.conv2_use_skip = config.conv2_use_skip
        self.output_features = config.pooled_features

    def _apply_position_bn(self, batch_norm: nn.BatchNorm1d, x: Tensor) -> Tensor:
        # x shape: (batch, channels=filters, length=post_conv1_length).
        # Transpose to (batch, length, channels) so `BatchNorm1d(post_conv1_length)` runs over the
        # length axis, matching Keras `kl.BatchNormalization(axis=1)`.
        return batch_norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, hidden_state: Tensor) -> Tensor:
        x1 = self.conv1(hidden_state)
        x1 = self.activation(x1)
        if self.use_batchnorm:
            x1 = self._apply_position_bn(self.batch_norm1, x1)
        x2 = self.conv2(x1)
        x2 = self.activation(x2)
        if self.use_batchnorm:
            x2 = self._apply_position_bn(self.batch_norm2, x2)
        if self.conv2_use_skip:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x2
        # Strided max-pool over the sequence dimension, then flatten in channels-last (Keras) order so
        # the dense layer sees the same element order as the original checkpoint.
        pooled = self.pool(x)
        # (batch, channels, length) -> (batch, length, channels) -> flatten.
        pooled = pooled.transpose(1, 2).reshape(pooled.size(0), -1)
        return pooled


class RbpEclipPositionModule(nn.Module):
    """GAM (B-spline) position module for the eight scalar genomic-landmark distance features.

    The upstream model pre-computes a B-spline basis of dimension `num_spline_bases` for each scalar
    distance feature, then applies a per-feature 1x1 convolution with `num_position_filters` outputs
    (named `conv_dist_<feature>`). For the default `as_track=False` configuration each feature is
    evaluated at a single position, so the GAM convolution reduces to a per-feature linear layer over
    the spline basis. The position-module output is the concatenation of the per-feature linear outputs.
    """

    def __init__(self, config: RbpEclipConfig):
        super().__init__()
        self.num_position_features = config.num_position_features
        self.num_position_filters = config.num_position_filters
        self.num_spline_bases = config.num_spline_bases
        self.feature_names = list(config.position_feature_names)
        self.linears = nn.ModuleList(
            [nn.Linear(config.num_spline_bases, config.num_position_filters) for _ in range(self.num_position_features)]
        )
        self.output_features = self.num_position_features * self.num_position_filters

    def forward(self, position_features: Tensor | None, batch_size: int) -> Tensor | None:
        if self.num_position_features == 0:
            return None
        if position_features is None:
            # The Kipoi default dataloader feeds zero spline-basis features when no transcript context is
            # available. Rebuild the deterministic zero tensor here rather than registering a buffer so
            # the model stays usable under transformers v5 meta-init.
            param = self.linears[0].weight
            position_features = torch.zeros(
                batch_size,
                self.num_position_features,
                self.num_spline_bases,
                device=param.device,
                dtype=param.dtype,
            )
        if position_features.dim() != 3 or position_features.size(1) != self.num_position_features:
            raise ValueError(
                "position_features must have shape "
                f"(batch_size, num_position_features={self.num_position_features}, "
                f"num_spline_bases={self.num_spline_bases}); got {tuple(position_features.shape)}."
            )
        if position_features.size(2) != self.num_spline_bases:
            raise ValueError(
                f"position_features last dimension ({position_features.size(2)}) must equal "
                f"num_spline_bases ({self.num_spline_bases})."
            )
        outputs = [
            self.linears[index](position_features[:, index, :].to(self.linears[index].weight.dtype))
            for index in range(self.num_position_features)
        ]
        return torch.cat(outputs, dim=-1)


class RbpEclipSplineTransform(nn.Module):
    r"""B-spline basis evaluator for raw scalar distance features.

    The upstream Kipoi `rbp_eclip` dataloader pre-computes the spline basis for every distance feature
    via `concise.preprocessing.encode_splines`; the converted MultiMolecule checkpoints follow the same
    convention and accept the basis directly as `position_features`. This helper provides a clean PyTorch
    implementation of the same basis evaluation so callers without the upstream preprocessing stack can
    transform raw distances into the basis representation.

    Args:
        num_bases: Number of B-spline basis functions (matches `RbpEclipConfig.num_spline_bases`).
        spline_degree: Polynomial degree of each basis spline. Defaults to 3 (cubic), as in
            `concise.preprocessing.encode_splines`.
        domain: Optional `(low, high)` range used to scale the inputs onto the basis support. When
            unspecified, the basis support is `[0, 1]` and the caller is expected to scale its inputs
            accordingly.
    """

    def __init__(
        self,
        num_bases: int,
        spline_degree: int = 3,
        domain: tuple[float, float] | None = None,
    ):
        super().__init__()
        if num_bases <= 0:
            raise ValueError(f"num_bases must be positive, but got {num_bases}.")
        if spline_degree < 0:
            raise ValueError(f"spline_degree must be non-negative, but got {spline_degree}.")
        self.num_bases = num_bases
        self.spline_degree = spline_degree
        self.domain = domain
        # Knot positions are deterministic from `num_bases` / `spline_degree`; rebuilt lazily in
        # `forward` rather than registered as a buffer to remain transformers v5 meta-init safe.

    @staticmethod
    def _bspline_basis(x: Tensor, knots: Tensor, degree: int) -> Tensor:
        """Evaluate the B-spline basis matrix at the locations ``x``.

        Implements the Cox-de Boor recursion in pure PyTorch.
        """
        num_bases = knots.size(0) - degree - 1
        # Degree-0 basis: indicator of each knot interval, with a closed right end-point on the last span
        # so the basis sums to one across `[knots[degree], knots[-degree - 1]]`.
        left = knots[:-1].unsqueeze(0)
        right = knots[1:].unsqueeze(0)
        x_expanded = x.unsqueeze(-1)
        basis = ((x_expanded >= left) & (x_expanded < right)).to(x.dtype)
        end = (x_expanded == knots[-1]).to(x.dtype)
        if basis.size(-1) > 0:
            basis[..., -1] = basis[..., -1] + end[..., 0]
            basis[..., -1] = basis[..., -1].clamp(max=1.0)
        for d in range(1, degree + 1):
            left_basis = basis[..., :-1]
            right_basis = basis[..., 1:]
            left_denom = knots[d:-1] - knots[: -d - 1]
            right_denom = knots[d + 1 :] - knots[1:-d]
            left_term = torch.where(
                left_denom != 0,
                (x_expanded - knots[: -d - 1]) / left_denom.clamp(min=torch.finfo(x.dtype).tiny),
                torch.zeros_like(left_basis),
            )
            right_term = torch.where(
                right_denom != 0,
                (knots[d + 1 :] - x_expanded) / right_denom.clamp(min=torch.finfo(x.dtype).tiny),
                torch.zeros_like(right_basis),
            )
            basis = left_term * left_basis + right_term * right_basis
        return basis[..., :num_bases]

    def forward(self, distances: Tensor) -> Tensor:
        if self.domain is not None:
            low, high = self.domain
            scaled = (distances.to(dtype=torch.float32) - low) / max(high - low, torch.finfo(torch.float32).tiny)
        else:
            scaled = distances.to(dtype=torch.float32)
        scaled = scaled.clamp(min=0.0, max=1.0)
        num_internal_knots = max(self.num_bases - self.spline_degree - 1, 0)
        if num_internal_knots > 0:
            internal = torch.linspace(0.0, 1.0, num_internal_knots + 2, device=scaled.device)[1:-1]
        else:
            internal = torch.empty(0, device=scaled.device)
        knots = torch.cat(
            [
                torch.zeros(self.spline_degree + 1, device=scaled.device),
                internal,
                torch.ones(self.spline_degree + 1, device=scaled.device),
            ]
        )
        return self._bspline_basis(scaled, knots, self.spline_degree).to(distances.dtype)


@dataclass
class RbpEclipModelOutput(ModelOutput):
    """
    Base class for outputs of the RBP-eCLIP model.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, num_hidden)`):
            The shared sequence + position representation consumed by the MultiMolecule sequence-prediction head.
        hidden_states:
            Always `None`; the RBP-eCLIP encoder collapses the sequence dimension through global max-pooling
            and exposes a single pooled feature vector. Provided for compatibility with the Transformers
            output convention.
        attentions:
            Always `None`; the RBP-eCLIP encoder is fully convolutional and has no attention layers.
            Provided for compatibility with the Transformers output convention.
    """

    pooler_output: Tensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None
