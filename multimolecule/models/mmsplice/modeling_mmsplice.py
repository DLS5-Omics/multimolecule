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
from transformers.utils.generic import can_return_tuple

from multimolecule.modules import Criterion

from ..modeling_outputs import SequencePredictorOutput
from .configuration_mmsplice import MODULE_ORDER, MmSpliceConfig, MmSpliceModuleConfig

# Upstream MMSplice clips the sigmoid splice-site probabilities before taking
# the logit (mmsplice.utils.logit, clip_threshold=1e-5).
LOGIT_CLIP = 1e-5


class MmSplicePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MmSpliceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["MmSpliceModule"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class MmSpliceModel(MmSplicePreTrainedModel):
    """
    The bare MMSplice modular backbone.

    MMSplice scores the exon-with-flanking-introns sequence with five independent
    sub-networks. The backbone returns the per-module score vector. For variant
    effect prediction, pass both a reference and an alternative sequence; the
    backbone then also returns the per-module score deltas.

    The five sub-networks do not share an architecture; each faithfully
    replicates the corresponding upstream
    [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice)
    Keras module (Cheng et al. 2019, *Genome Biology*).

    Examples:
        >>> import torch
        >>> from multimolecule import MmSpliceConfig, MmSpliceModel
        >>> config = MmSpliceConfig()
        >>> model = MmSpliceModel(config)
        >>> _ = model.eval()
        >>> input_ids = torch.randint(4, (1, 400))
        >>> output = model(input_ids)
        >>> output["logits"].shape
        torch.Size([1, 5])
    """

    def __init__(self, config: MmSpliceConfig):
        super().__init__(config)
        self.embeddings = MmSpliceEmbedding(config)
        self.region_models = nn.ModuleDict({name: MmSpliceModule(config, name) for name in MODULE_ORDER})
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MmSpliceModelOutput | tuple[Tensor, ...]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        reference = self._score(input_ids, attention_mask, inputs_embeds)

        delta = None
        alternative = None
        has_alternative = alternative_input_ids is not None or alternative_inputs_embeds is not None
        if has_alternative:
            if alternative_input_ids is not None and alternative_inputs_embeds is not None:
                raise ValueError("You cannot specify both alternative_input_ids and alternative_inputs_embeds")
            alternative = self._score(
                alternative_input_ids,
                alternative_attention_mask,
                alternative_inputs_embeds,
            )
            delta = alternative - reference

        return MmSpliceModelOutput(
            logits=reference,
            alternative_logits=alternative,
            delta_logits=delta,
        )

    def _score(
        self,
        input_ids: Tensor | NestedTensor | None,
        attention_mask: Tensor | None,
        inputs_embeds: Tensor | NestedTensor | None,
    ) -> Tensor:
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
        scores = []
        for name in MODULE_ORDER:
            module = self.region_models[name]
            if self.gradient_checkpointing and self.training:
                score = self._gradient_checkpointing_func(module.__call__, embedding_output)
            else:
                score = module(embedding_output)
            scores.append(score)
        return torch.cat(scores, dim=-1)


class MmSpliceForSequencePrediction(MmSplicePreTrainedModel):
    """
    MMSplice with a sequence-level prediction head.

    The head consumes the per-module score deltas for a reference/alternative
    sequence pair and applies the fixed upstream linear combiner to produce the
    delta-logit-PSI splicing-effect score.

    Examples:
        >>> import torch
        >>> from multimolecule import MmSpliceConfig, MmSpliceForSequencePrediction
        >>> config = MmSpliceConfig()
        >>> model = MmSpliceForSequencePrediction(config)
        >>> _ = model.eval()
        >>> input_ids = torch.randint(4, (1, 400))
        >>> alternative_input_ids = torch.randint(4, (1, 400))
        >>> output = model(input_ids, alternative_input_ids=alternative_input_ids)
        >>> output["logits"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: MmSpliceConfig):
        super().__init__(config)
        self.model = MmSpliceModel(config)
        self.prediction = MmSpliceDeltaLogitPsiHead()
        head = config.head
        if head is None:
            raise ValueError("MmSpliceForSequencePrediction requires `config.head` to be set")
        self.criterion = Criterion(head)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        alternative_input_ids: Tensor | NestedTensor | None = None,
        alternative_attention_mask: Tensor | None = None,
        alternative_inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, ...] | SequencePredictorOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            alternative_input_ids=alternative_input_ids,
            alternative_attention_mask=alternative_attention_mask,
            alternative_inputs_embeds=alternative_inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        if outputs.delta_logits is None:
            raise ValueError(
                "MmSpliceForSequencePrediction requires an alternative sequence to compute delta-logit-PSI. "
                "Use MmSpliceModel for reference-only module scores."
            )
        logits = self.prediction(outputs.delta_logits)
        loss = self.criterion(logits, labels) if labels is not None else None

        return SequencePredictorOutput(loss=loss, logits=logits)


class MmSpliceModule(nn.Module):
    """
    A single MMSplice sub-network scoring one region of the input sequence.

    Dispatches to the `conv` or `dense` architecture family depending on the
    module configuration. Each module first slices its genomic region from the
    one-hot encoded sequence (replicating the upstream ``SeqSpliter``), then
    applies its sub-network, producing a single scalar score per sequence.
    """

    def __init__(self, config: MmSpliceConfig, name: str):
        super().__init__()
        self.name = name
        module_config: MmSpliceModuleConfig = config.modules_config[name]
        self.architecture = module_config["architecture"]
        self.region = MmSpliceRegion(config, name)
        if self.architecture == "conv":
            self.network = MmSpliceConvModule(config, module_config)
            self.logit = False
        elif self.architecture == "dense":
            self.network = MmSpliceDenseModule(config, module_config)
            # Upstream applies logit() to the sigmoid splice-site probabilities.
            self.logit = True
        else:
            raise ValueError(f"Unknown MMSplice module architecture '{self.architecture}'.")

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_state = self.region(inputs_embeds)
        score = self.network(hidden_state)
        if self.logit:
            score = score.clamp(LOGIT_CLIP, 1 - LOGIT_CLIP)
            score = torch.log(score) - torch.log1p(-score)
        return score


class MmSpliceEmbedding(nn.Module):
    """
    One-hot encodes the input sequence into the upstream MMSplice channel order.

    Produces a ``(batch, vocab_size, length)`` tensor; downstream modules slice
    their region along the length dimension. Tokens outside the nucleobase
    alphabet (e.g. the ``N`` padding token) map to an all-zero column, matching
    the upstream ``encodeDNA`` behaviour.
    """

    def __init__(self, config: MmSpliceConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.register_buffer("_float_tensor", torch.empty(0), persistent=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            channel_ids = input_ids.to(torch.long)
            valid = (channel_ids >= 0) & (channel_ids < self.vocab_size)
            inputs_embeds = torch.zeros(
                (*channel_ids.shape, self.vocab_size),
                dtype=self._float_tensor.dtype,
                device=channel_ids.device,
            )
            inputs_embeds.scatter_(
                -1,
                channel_ids.clamp(min=0, max=self.vocab_size - 1).unsqueeze(-1),
                valid.to(dtype=inputs_embeds.dtype).unsqueeze(-1),
            )
        else:
            inputs_embeds = inputs_embeds.to(dtype=self._float_tensor.dtype)
        if attention_mask is not None:
            inputs_embeds = (inputs_embeds * attention_mask.unsqueeze(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.transpose(1, 2)


class MmSpliceRegion(nn.Module):
    """
    Slices the region of the sequence consumed by a given MMSplice module.

    Replicates the upstream ``mmsplice.exon_dataloader.SeqSpliter.split``. The
    MultiMolecule input is the exon with symmetric ``overhang`` flanking introns
    (default 100 bp each side). The acceptor / donor splice-site modules consume
    a fixed-length window; when the sequence is shorter than required the missing
    intronic context is left-/right-padded with all-zero columns (the upstream
    ``N`` padding).
    """

    def __init__(self, config: MmSpliceConfig, name: str):
        super().__init__()
        self.name = name
        self.overhang = (100, 100)
        self.acceptor_intron_cut = config.acceptor_intron_cut
        self.donor_intron_cut = config.donor_intron_cut
        self.acceptor_intron_length = config.acceptor_intron_length
        self.acceptor_exon_length = config.acceptor_exon_length
        self.donor_exon_length = config.donor_exon_length
        self.donor_intron_length = config.donor_intron_length

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        intronl, intronr = self.overhang

        # MMSplice operates on an exon flanked by `overhang` intron bp on each
        # side. The MultiMolecule interface only sees the token sequence, so the
        # whole input is interpreted as the overhanged exon. When the sequence is
        # shorter than the two flanks (degenerate / unit-test inputs) the missing
        # context is zero-padded, mirroring the upstream ``SeqSpliter`` N-pad and
        # keeping the slice arithmetic valid (a no-op for realistic inputs).
        minimum = intronl + intronr + 1
        if inputs_embeds.size(-1) < minimum:
            inputs_embeds = F.pad(inputs_embeds, (0, minimum - inputs_embeds.size(-1)))
        length = inputs_embeds.size(-1)

        # Upstream pads N if the available intron context is shorter than the
        # length the splice-site models were trained on.
        lackl = self.acceptor_intron_length - intronl
        padl = 0
        if lackl >= 0:
            padl = lackl + 1
            intronl += padl
        lackr = self.donor_intron_length - intronr
        padr = 0
        if lackr >= 0:
            padr = lackr + 1
            intronr += padr
        if padl or padr:
            inputs_embeds = F.pad(inputs_embeds, (padl, padr))
            length = inputs_embeds.size(-1)

        if self.name == "acceptor_intron":
            return inputs_embeds[..., : intronl - self.acceptor_intron_cut]
        if self.name == "acceptor":
            start = intronl - self.acceptor_intron_length
            end = intronl + self.acceptor_exon_length
            return self._window(inputs_embeds, start, end)
        if self.name == "exon":
            exon = inputs_embeds[..., intronl : length - intronr]
            if exon.size(-1) == 0:
                exon = inputs_embeds.new_zeros(inputs_embeds.size(0), inputs_embeds.size(1), 1)
            return exon
        if self.name == "donor":
            start = length - intronr - self.donor_exon_length
            end = length - intronr + self.donor_intron_length
            return self._window(inputs_embeds, start, end)
        # donor_intron
        return inputs_embeds[..., length - intronr + self.donor_intron_cut :]

    @staticmethod
    def _window(inputs_embeds: Tensor, start: int, end: int) -> Tensor:
        """Fixed-length splice-site window, zero-padded (upstream ``N`` pad) when
        the sequence is shorter than the configured region."""
        length = inputs_embeds.size(-1)
        pad_left = max(0, -start)
        pad_right = max(0, end - length)
        window = inputs_embeds[..., max(0, start) : min(length, end)]
        if pad_left or pad_right:
            window = F.pad(window, (pad_left, pad_right))
        return window


class MmSpliceConvModule(nn.Module):
    """
    Length-independent convolutional MMSplice sub-network.

    Replicates the upstream ``Intron3`` / ``Exon`` / ``Intron5`` modules:
    a single length-preserving 1D convolution, an optional batch-norm, a ReLU,
    global average pooling over positions, and a linear projection to a scalar.
    """

    def __init__(self, config: MmSpliceConfig, module_config: MmSpliceModuleConfig):
        super().__init__()
        channels = module_config["conv_channels"]
        self.pool_mask_zeros = module_config["pool_mask_zeros"]
        self.conv = nn.Conv1d(
            config.vocab_size,
            channels,
            kernel_size=module_config["conv_kernel_size"],
            padding="same",
        )
        self.norm = (
            nn.BatchNorm1d(channels, eps=module_config["batch_norm_eps"]) if module_config["conv_batch_norm"] else None
        )
        self.activation = ACT2FN[module_config["conv_activation"]]
        self.decoder = nn.Linear(channels, 1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        mask = hidden_state.amax(dim=1, keepdim=True) if self.pool_mask_zeros else None
        hidden_state = self.conv(hidden_state)
        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        if mask is None:
            pooled = hidden_state.mean(dim=-1)
        else:
            hidden_state = hidden_state * mask
            denominator = mask.sum(dim=-1).clamp_min(torch.finfo(hidden_state.dtype).eps)
            pooled = hidden_state.sum(dim=-1) / denominator
        return self.decoder(pooled)


class MmSpliceDenseModule(nn.Module):
    """
    Fixed-length splice-site MMSplice sub-network.

    Replicates the upstream ``Acceptor`` / ``Donor`` modules. The fixed-length
    region is optionally passed through a length-preserving convolution, a
    ``1x1`` convolution (each followed by a batch-norm), then flattened in
    Keras channel-last order and projected to a probability with a stack of
    dense + batch-norm + ReLU + dropout blocks and a final sigmoid.
    """

    def __init__(self, config: MmSpliceConfig, module_config: MmSpliceModuleConfig):
        super().__init__()
        region_length = module_config["region_length"]
        eps = module_config["batch_norm_eps"]
        channels = config.vocab_size

        self.conv = None
        self.conv_norm = None
        self.conv_activation = ACT2FN[module_config["conv_activation"]]
        if module_config["conv_channels"]:
            channels = module_config["conv_channels"]
            self.conv = nn.Conv1d(
                config.vocab_size,
                channels,
                kernel_size=module_config["conv_kernel_size"],
                padding="same",
            )
            self.conv_norm = nn.BatchNorm1d(channels, eps=eps) if module_config["conv_batch_norm"] else None

        self.pointwise = None
        self.pointwise_norm = None
        if module_config["pointwise_channels"]:
            pw = module_config["pointwise_channels"]
            self.pointwise = nn.Conv1d(channels, pw, kernel_size=1)
            self.pointwise_norm = nn.BatchNorm1d(pw, eps=eps)
            channels = pw

        self.flatten_dropout = nn.Dropout(module_config["dropout"]) if module_config["flatten_dropout"] else None

        blocks = []
        in_features = region_length * channels
        for hidden_size in module_config["hidden_sizes"]:
            blocks.append(MmSpliceDenseBlock(in_features, hidden_size, module_config["dropout"], eps))
            in_features = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.decoder = nn.Linear(in_features, 1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        if self.conv is not None:
            hidden_state = self.conv_activation(self.conv(hidden_state))
            if self.conv_norm is not None:
                hidden_state = self.conv_norm(hidden_state)
        if self.pointwise is not None:
            hidden_state = F.relu(self.pointwise(hidden_state))
            if self.pointwise_norm is None:
                raise RuntimeError("MMSplice pointwise convolution is missing its batch normalization layer")
            hidden_state = self.pointwise_norm(hidden_state)
        # Keras Flatten on a (batch, length, channels) tensor; our tensors are
        # (batch, channels, length), so transpose to match the flatten order.
        hidden_state = hidden_state.transpose(1, 2).reshape(hidden_state.size(0), -1)
        if self.flatten_dropout is not None:
            hidden_state = self.flatten_dropout(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return torch.sigmoid(self.decoder(hidden_state))


class MmSpliceDenseBlock(nn.Module):
    """Dense + batch-norm + ReLU + dropout block of a `dense`-family head."""

    def __init__(self, in_features: int, out_features: int, dropout: float, eps: float):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = F.relu(hidden_state)
        return self.dropout(hidden_state)


class MmSpliceDeltaLogitPsiHead(nn.Module):
    """
    Fixed upstream MMSplice linear combiner for delta-logit-PSI prediction.
    """

    def __init__(self):
        super().__init__()

    def forward(self, delta_logits: Tensor) -> Tensor:
        zeros = torch.zeros((), device=delta_logits.device, dtype=delta_logits.dtype)
        not_close = ~torch.isclose(delta_logits, zeros)
        exon_overlap = (not_close[:, 1] & not_close[:, 2]) | (not_close[:, 2] & not_close[:, 3])
        acceptor_intron_overlap = not_close[:, 0] & not_close[:, 1]
        donor_intron_overlap = not_close[:, 3] & not_close[:, 4]
        features = torch.cat(
            [
                delta_logits,
                (delta_logits[:, 2] * exon_overlap.to(delta_logits.dtype)).unsqueeze(-1),
                (delta_logits[:, 4] * donor_intron_overlap.to(delta_logits.dtype)).unsqueeze(-1),
                (delta_logits[:, 0] * acceptor_intron_overlap.to(delta_logits.dtype)).unsqueeze(-1),
            ],
            dim=-1,
        )
        coefficients = delta_logits.new_tensor(
            [
                0.49685773,
                0.72322957,
                1.54760024,
                0.75011527,
                2.26187717,
                -0.69419094,
                2.40138709,
                0.88148553,
            ]
        )
        intercept = delta_logits.new_tensor(0.0006480262366686865)
        return features.matmul(coefficients.unsqueeze(-1)) + intercept


@dataclass
class MmSpliceModelOutput(ModelOutput):
    """
    Base class for outputs of the MMSplice modular model.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The per-module score vector for the (reference) input sequence. The
            module order is `acceptor_intron`, `acceptor`, `exon`, `donor`,
            `donor_intron`.
        alternative_logits (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            The per-module score vector for the alternative sequence, returned when
            an alternative sequence is provided.
        delta_logits (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            `alternative_logits - logits`, the per-module variant-effect deltas,
            returned when an alternative sequence is provided.
    """

    logits: torch.FloatTensor | None = None
    alternative_logits: torch.FloatTensor | None = None
    delta_logits: torch.FloatTensor | None = None
