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
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import Criterion, HeadConfig

from ..modeling_outputs import SequencePredictorOutput
from .configuration_framepool import FramepoolConfig


class FramepoolPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FramepoolConfig
    base_model_prefix = "model"
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["FramepoolEncoder"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)


class FramepoolModel(FramepoolPreTrainedModel):
    """
    The bare Framepool model, producing a frame-aware representation from a 5'UTR sequence.

    Framepool replaces the fixed-length flatten of [Sample et al., 2019](https://doi.org/10.1038/s41587-019-0164-5)
    with a frame-aware pooling layer that splits the convolutional feature map into the three reading frames relative
    to the start codon and pools each frame independently. The resulting representation is length-independent.

    Examples:
        >>> from multimolecule import FramepoolConfig, FramepoolModel, RnaTokenizer
        >>> config = FramepoolConfig()
        >>> model = FramepoolModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/framepool")
        >>> input = tokenizer("ACGUACGUACGU", return_tensors="pt")
        >>> output = model(**input)
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    def __init__(self, config: FramepoolConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = FramepoolEmbedding(config)
        self.encoder = FramepoolEncoder(config)
        self.pooler = FramepoolPooler(config)
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
    ) -> FramepoolModelOutput:
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

        # ``(batch, vocab_size, length)``; padding tokens (and tokens outside the nucleobase alphabet)
        # are encoded as all-zero columns.
        embedding_output, pad_mask = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = self.encoder(embedding_output, pad_mask)
        pooled_output = self.pooler(hidden_state, pad_mask)

        return FramepoolModelOutput(
            pooler_output=pooled_output,
            last_hidden_state=hidden_state.transpose(1, 2),
        )


class FramepoolForSequencePrediction(FramepoolPreTrainedModel):
    """
    Framepool with a sequence-level prediction head.

    When called with a single sequence the head returns the unscaled mean ribosome load (MRL) prediction. When called
    with both a reference and an alternative sequence it returns the ``log2`` mean ribosome load fold change
    (``log2(alternative / reference)``), matching the upstream Kipoi
    ``UTRVariantEffectModel`` variant effect interface.

    Examples:
        >>> import torch
        >>> from multimolecule import FramepoolConfig, FramepoolForSequencePrediction, RnaTokenizer
        >>> config = FramepoolConfig()
        >>> model = FramepoolForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/framepool")
        >>> input = tokenizer("ACGUACGUACGU", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1.0]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> alternative = tokenizer("ACGUACGUACGA", return_tensors="pt")
        >>> output = model(**input, alternative_input_ids=alternative["input_ids"])
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: FramepoolConfig):
        super().__init__(config)
        self.model = FramepoolModel(config)
        head_config = HeadConfig(config.head) if config.head is not None else HeadConfig()
        if head_config.num_labels is None:
            head_config.num_labels = config.num_labels
        if head_config.problem_type is None:
            head_config.problem_type = "regression"
        self.head_config = head_config
        self.criterion = Criterion(head_config)
        self.prediction = FramepoolMrlHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def output_channels(self) -> list[str]:
        return ["mean_ribosome_load"]

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        library_indicator: Tensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | SequencePredictorOutput:
        reference = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        reference_logits = self.prediction(reference.pooler_output, library_indicator=library_indicator)

        has_alternative = alternative_input_ids is not None or alternative_inputs_embeds is not None
        if has_alternative:
            alternative = self.model(
                alternative_input_ids,
                attention_mask=alternative_attention_mask,
                inputs_embeds=alternative_inputs_embeds,
                return_dict=True,
                **kwargs,
            )
            alternative_logits = self.prediction(alternative.pooler_output, library_indicator=library_indicator)
            # ``log2(alt / ref)`` matches the Kipoi `UTRVariantEffectModel.predict_on_batch` MRL fold-change output.
            logits = torch.log2(alternative_logits / reference_logits)
            loss = self.criterion(logits, labels) if labels is not None else None
            return SequencePredictorOutput(loss=loss, logits=logits)

        logits = reference_logits
        loss = self.criterion(logits, labels) if labels is not None else None
        return SequencePredictorOutput(loss=loss, logits=logits)


class FramepoolEmbedding(nn.Module):
    """One-hot embedding that derives input channels from the MultiMolecule token order."""

    def __init__(self, config: FramepoolConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.null_channel_id = config.null_channel_id
        # Zero-size buffer used to track the model's current dtype after .half() / .to(bf16)
        # so F.one_hot output (always int64) can be cast to the active dtype in forward.
        self.register_buffer("_dtype_reference", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        dtype = self._dtype_reference.dtype
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            valid = (input_ids >= 0) & (input_ids < self.vocab_size)
            inputs_embeds = F.one_hot(input_ids.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).to(dtype=dtype)
            if not valid.all():
                inputs_embeds = inputs_embeds * valid.unsqueeze(-1).to(dtype)
        else:
            inputs_embeds = inputs_embeds.to(dtype=dtype)
        if self.null_channel_id is not None:
            # Zero out the trailing "no nucleobase" column so that the padding-mask sum below correctly identifies
            # padded positions, matching the upstream ``nuc_dict["N"] = [0, 0, 0, 0]`` convention.
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[..., self.null_channel_id] = 0
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        # Padding mask derived from the channel sum, matching the upstream `compute_pad_mask` (Keras model.py).
        # Real tokens contribute one to the mask, all-zero columns (N / padding / special tokens) contribute zero.
        pad_mask = inputs_embeds.sum(dim=-1)
        # (batch, vocab_size, length) for the 1D convolutional stack.
        return inputs_embeds.transpose(1, 2), pad_mask


class FramepoolEncoder(nn.Module):
    """Stack of residual length-preserving 1D convolutions with masked activations."""

    def __init__(self, config: FramepoolConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = config.skip_connections
        in_channels = config.vocab_size
        for index in range(config.num_conv_layers):
            self.layers.append(
                FramepoolConvLayer(
                    in_channels=in_channels,
                    out_channels=config.num_filters,
                    kernel_size=config.kernel_size[index],
                    dilation=config.dilations[index],
                    padding=config.padding,
                    activation=config.hidden_act,
                )
            )
            in_channels = config.num_filters

    def forward(self, hidden_state: Tensor, pad_mask: Tensor) -> Tensor:
        # ``mask`` is broadcast across the channel axis to zero out padded positions, mirroring
        # the upstream ``apply_pad_mask`` Lambda that wraps every convolution.
        mask = pad_mask.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            residual = hidden_state
            hidden_state = layer(hidden_state, mask)
            if self.skip_connections == "residual" and index > 0:
                hidden_state = hidden_state + residual
        return hidden_state


class FramepoolConvLayer(nn.Module):
    """A single length-preserving 1D convolution followed by masking and a non-linearity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: str,
        activation: str,
    ):
        super().__init__()
        # ``same`` padding in Keras pads the right end of the sequence by one extra position when the
        # effective kernel is even; PyTorch's ``padding="same"`` follows the identical convention.
        self.padding = padding
        if padding == "causal":
            self.causal_pad = dilation * (kernel_size - 1)
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        else:
            self.causal_pad = 0
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        if self.padding == "causal":
            hidden_state = F.pad(hidden_state, (self.causal_pad, 0))
        hidden_state = self.conv(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state * mask


class FramepoolPooler(nn.Module):
    """Frame-aware pooling that reverses the sequence, slices each reading frame, and pools per frame."""

    def __init__(self, config: FramepoolConfig):
        super().__init__()
        self.only_max_pool = config.only_max_pool

    def forward(self, hidden_state: Tensor, pad_mask: Tensor) -> Tensor:
        # Reverse along the length axis so that frame membership is anchored to the canonical start codon
        # (the right-most position of the upstream-encoded 5'UTR), matching the upstream ``FrameSliceLayer``.
        reversed_features = torch.flip(hidden_state, dims=(-1,))
        reversed_mask = torch.flip(pad_mask, dims=(-1,))
        pooled = []
        for offset in range(3):
            # ``frame_features``: (batch, channels, frame_length).
            frame_features = reversed_features[..., offset::3]
            # Global max pool over positions; tokens are non-negative after the encoder's ReLU + mask, so
            # zeroed positions never beat real activations for the max.
            pooled.append(frame_features.amax(dim=-1))
        if not self.only_max_pool:
            eps = torch.finfo(reversed_features.dtype).eps
            # The released checkpoint wires the unsliced pad mask into every average-pooling call, so each
            # frame sum is divided by the total valid sequence length rather than the per-frame length.
            denominator = reversed_mask.sum(dim=-1, keepdim=True).clamp_min(eps)
            for offset in range(3):
                frame_features = reversed_features[..., offset::3]
                pooled.append(frame_features.sum(dim=-1) / denominator)
        return torch.cat(pooled, dim=-1)


class FramepoolMrlHead(nn.Module):
    """Dense stack + scaling regression producing the (sub-library aware) mean ribosome load prediction."""

    def __init__(self, config: FramepoolConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = config.hidden_size
        for hidden_size in config.dense_sizes:
            layers.append(FramepoolDenseBlock(in_features, hidden_size, config.dense_dropout, config.hidden_act))
            in_features = hidden_size
        self.dense = nn.ModuleList(layers)
        self.unscaled = nn.Linear(in_features, 1)
        # Scaling regression: an input of dimension ``2 * library_size`` (the per-library copies of the unscaled
        # prediction concatenated with the one-hot library indicator) is projected to a single scalar, without bias.
        self.library_size = config.library_size
        self.library_index = config.library_index
        self.scaling = nn.Linear(2 * config.library_size, 1, bias=False)

    def forward(self, hidden_state: Tensor, library_indicator: Tensor | None = None) -> Tensor:
        for layer in self.dense:
            hidden_state = layer(hidden_state)
        unscaled = self.unscaled(hidden_state)
        if library_indicator is None:
            # Deterministic constant one-hot library indicator, rebuilt every forward; the released checkpoint
            # uses the ``random`` library (index 1) for variant effect prediction (Kipoi ``UTRVariantEffectModel``).
            library_indicator = torch.zeros(
                hidden_state.size(0),
                self.library_size,
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )
            library_indicator[:, self.library_index] = 1.0
        else:
            library_indicator = library_indicator.to(device=hidden_state.device, dtype=hidden_state.dtype)
            if library_indicator.shape != (hidden_state.size(0), self.library_size):
                raise ValueError(
                    f"library_indicator must have shape ({hidden_state.size(0)}, {self.library_size}), "
                    f"got {tuple(library_indicator.shape)}."
                )
        # Interaction term: per-library copies of the scalar prediction.
        interaction = unscaled * library_indicator
        regression_features = torch.cat([interaction, library_indicator], dim=-1)
        return self.scaling(regression_features)


class FramepoolDenseBlock(nn.Module):
    """Dense + activation + dropout block of the prediction head."""

    def __init__(self, in_features: int, out_features: int, dropout: float, activation: str):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = ACT2FN[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.dropout(self.activation(self.dense(hidden_state)))


@dataclass
class FramepoolModelOutput(ModelOutput):
    """
    Base class for outputs of the Framepool model.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The concatenation of per-frame max (and optionally average) pooled feature vectors consumed by the
            sequence-level prediction head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_filters)`):
            The encoder feature map before the frame-aware pooling, with padded positions zeroed out.
    """

    pooler_output: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
