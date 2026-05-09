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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import preserve_batch_norm_stats

from .configuration_ufold import UfoldConfig


class UfoldPreTrainedModel(PreTrainedModel):

    config_class = UfoldConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["UfoldConvBlock"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        # Use transformers.initialization wrappers (imported as `init`); they check the
        # `_is_hf_initialized` flag so they don't clobber tensors loaded from a checkpoint.
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class UfoldModel(UfoldPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import UfoldConfig, UfoldModel
        >>> config = UfoldConfig(postprocess_iterations=1)
        >>> model = UfoldModel(config)
        >>> input_ids = torch.tensor([[0, 3, 2, 1, 0, 3]])
        >>> output = model(input_ids=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 6, 6])
        >>> output["contact_map"].shape
        torch.Size([1, 6, 6])
    """

    def __init__(self, config: UfoldConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.encoder = UfoldEncoder(
            config.input_channels,
            config.output_channels,
            config.channel_sizes,
            batch_norm_eps=config.batch_norm_eps,
            batch_norm_momentum=config.batch_norm_momentum,
        )
        self.supports_batch_process = True
        self.register_buffer(
            "pair_score",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 3.0, 0.0, 0.8],
                    [2.0, 0.0, 0.8, 0.0],
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "gaussian_weights",
            torch.exp(-0.5 * torch.arange(30, dtype=torch.float32).pow(2)),
            persistent=False,
        )
        self.register_buffer("pos_weight", torch.tensor([config.pos_weight]), persistent=False)
        self.post_init()

    def postprocess(self, outputs, input_ids=None, **kwargs):
        postprocessed_contact_map = outputs.get("postprocessed_contact_map")
        if postprocessed_contact_map is not None:
            return postprocessed_contact_map
        return outputs["contact_map"]

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        use_postprocessing: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> UfoldModelOutput:
        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is not None and isinstance(inputs_embeds, NestedTensor):
            raise TypeError("UfoldModel does not support NestedTensor inputs_embeds")

        inputs_embeds = self._prepare_inputs_embeds(input_ids, attention_mask, inputs_embeds)
        lengths = _get_lengths(inputs_embeds, attention_mask)
        inputs_embeds = self._pad_inputs_embeds(inputs_embeds, lengths)

        features = ufold_features(inputs_embeds, self.pair_score, self.gaussian_weights)
        logits_padded = self.encoder(features)
        max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
        logits = _crop_batch(logits_padded, lengths, max_length)

        should_postprocess = self.config.use_postprocessing if use_postprocessing is None else use_postprocessing
        postprocessed_contact_map = None
        if should_postprocess:
            with torch.no_grad():
                postprocessed_contact_map_padded = self._postprocess(logits_padded, inputs_embeds)
            postprocessed_contact_map = _crop_batch(postprocessed_contact_map_padded, lengths, max_length)
            contact_map = postprocessed_contact_map
        else:
            contact_map = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            labels = labels.to(device=logits.device, dtype=logits.dtype)
            labels = labels[:, :max_length, :max_length]
            valid_mask = _pair_mask(lengths, max_length, logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits[valid_mask], labels[valid_mask], pos_weight=self.pos_weight
            )

        return UfoldModelOutput(
            loss=loss,
            logits=logits,
            contact_map=contact_map,
            postprocessed_contact_map=postprocessed_contact_map,
        )

    def _prepare_inputs_embeds(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if inputs_embeds is not None:
            if inputs_embeds.size(-1) < 4:
                raise ValueError(f"inputs_embeds last dimension ({inputs_embeds.size(-1)}) must be at least 4.")
            one_hot = inputs_embeds[..., :4].to(dtype=self.pair_score.dtype)
        else:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_size)[..., :4].to(dtype=self.pair_score.dtype)

        if attention_mask is not None:
            one_hot = one_hot * attention_mask.unsqueeze(-1).to(one_hot.dtype)
        return one_hot

    def _pad_inputs_embeds(self, inputs_embeds: Tensor, lengths: Tensor | None = None) -> Tensor:
        if lengths is None:
            max_length = inputs_embeds.size(1)
        else:
            max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
        padded_length = _get_padded_length(
            max_length,
            min_length=self.config.min_size,
            multiple=self.config.size_multiple,
        )
        return _fit_length(inputs_embeds, padded_length)

    def _postprocess(self, logits: Tensor, one_hot: Tensor) -> Tensor:
        mask = _constraint_matrix(one_hot, allow_noncanonical=self.config.allow_noncanonical).to(dtype=logits.dtype)
        u = torch.sigmoid(2 * (logits - self.config.postprocess_s)) * logits
        a_hat = torch.sigmoid(u) * torch.sigmoid(2 * (u - self.config.postprocess_s)).detach()
        lmbd = F.relu(_contact_a(a_hat, mask).sum(dim=-1) - 1).detach()

        lr_min = self.config.postprocess_lr_min
        lr_max = self.config.postprocess_lr_max
        for _ in range(self.config.postprocess_iterations):
            violation = torch.sigmoid(2 * (_contact_a(a_hat, mask).sum(dim=-1) - 1))
            grad_a = (lmbd * violation).unsqueeze(-1).expand_as(logits) - u / 2
            grad = a_hat * mask * (grad_a + grad_a.transpose(-1, -2))
            a_hat = a_hat - lr_min * grad
            lr_min *= 0.99

            if self.config.postprocess_with_l1:
                a_hat = F.relu(torch.abs(a_hat) - self.config.postprocess_rho * lr_min)

            lmbd_grad = F.relu(_contact_a(a_hat, mask).sum(dim=-1) - 1)
            lmbd = lmbd + lr_max * lmbd_grad
            lr_max *= 0.99

        return _contact_a(a_hat, mask)


class UfoldForRnaSecondaryStructurePrediction(UfoldModel):
    pass


class UfoldEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 17,
        output_channels: int = 1,
        channels: list[int] | None = None,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        channels = [32, 64, 128, 256, 512] if channels is None else channels

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_blocks = nn.ModuleList(
            [
                UfoldConvBlock(
                    input_channels if block_index == 0 else channels[block_index - 1],
                    output_channels,
                    batch_norm_eps,
                    batch_norm_momentum,
                )
                for block_index, output_channels in enumerate(channels)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UfoldUpBlock(input_channels, output_channels, batch_norm_eps, batch_norm_momentum)
                for input_channels, output_channels in zip(reversed(channels[1:]), reversed(channels[:-1]))
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                UfoldConvBlock(input_channels, output_channels, batch_norm_eps, batch_norm_momentum)
                for input_channels, output_channels in zip(
                    [2 * channels[index] for index in range(len(channels) - 2, -1, -1)],
                    reversed(channels[:-1]),
                )
            ]
        )
        self.prediction = nn.Conv2d(channels[0], output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual_states = []
        for block_index, block in enumerate(self.down_blocks):
            if block_index > 0:
                hidden_states = self.pooling(hidden_states)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    context_fn=lambda block=block: (nullcontext(), preserve_batch_norm_stats(block)),
                )
            else:
                hidden_states = block(hidden_states)
            residual_states.append(hidden_states)

        hidden_states = residual_states[-1]
        for up_block, decoder_block, residual_state in zip(
            self.up_blocks,
            self.decoder_blocks,
            reversed(residual_states[:-1]),
        ):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    up_block.__call__,
                    hidden_states,
                    context_fn=lambda up_block=up_block: (nullcontext(), preserve_batch_norm_stats(up_block)),
                )
            else:
                hidden_states = up_block(hidden_states)
            hidden_states = torch.cat((residual_state, hidden_states), dim=1)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_block.__call__,
                    hidden_states,
                    context_fn=lambda decoder_block=decoder_block: (
                        nullcontext(),
                        preserve_batch_norm_stats(decoder_block),
                    ),
                )
            else:
                hidden_states = decoder_block(hidden_states)

        logits = self.prediction(hidden_states).squeeze(1)
        return logits.transpose(-1, -2) * logits


class UfoldConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.activation2 = nn.ReLU(inplace=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.batch_norm1(hidden_states)
        hidden_states = self.activation1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.batch_norm2(hidden_states)
        return self.activation2(hidden_states)


class UfoldUpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.upsampling(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        return self.activation(hidden_states)


def ufold_features(
    inputs_embeds: Tensor,
    pair_score: Tensor,
    gaussian_weights: Tensor | None = None,
) -> Tensor:
    """
    Build the original 17-channel UFold image representation.

    The first 16 channels follow `itertools.product(range(4), range(4))` over the tokenizer nucleotide order
    `A, C, G, U`. The final channel is the hand-crafted paired-neighbor prior used by the original code.
    """
    pairwise = inputs_embeds.unsqueeze(2).unsqueeze(-1) * inputs_embeds.unsqueeze(1).unsqueeze(-2)
    pairwise = pairwise.flatten(start_dim=-2).permute(0, 3, 1, 2)
    prior = ufold_pairing_prior(inputs_embeds, pair_score, gaussian_weights).unsqueeze(1)
    return torch.cat([pairwise, prior], dim=1)


def ufold_pairing_prior(
    inputs_embeds: Tensor,
    pair_score: Tensor,
    gaussian_weights: Tensor | None = None,
) -> Tensor:
    base_scores = torch.einsum("bia,ac,bjc->bij", inputs_embeds, pair_score.to(inputs_embeds), inputs_embeds)
    batch_size, length, _ = base_scores.shape
    prior = base_scores.new_zeros(batch_size, length, length)

    if gaussian_weights is None:
        gaussian_weights = torch.exp(-0.5 * torch.arange(30, device=base_scores.device, dtype=base_scores.dtype).pow(2))
    else:
        gaussian_weights = gaussian_weights.to(base_scores)

    alive = torch.ones_like(base_scores, dtype=torch.bool)
    for offset in range(min(30, length)):
        shifted = base_scores.new_zeros(batch_size, length, length)
        shifted[:, offset:, : length - offset] = base_scores[:, : length - offset, offset:]
        active = alive & shifted.ne(0)
        prior = prior + shifted * gaussian_weights[offset] * active.to(shifted.dtype)
        alive = active

    alive = prior.gt(0)
    for offset in range(1, min(30, length)):
        shifted = base_scores.new_zeros(batch_size, length, length)
        shifted[:, : length - offset, offset:] = base_scores[:, offset:, : length - offset]
        active = alive & shifted.ne(0)
        prior = prior + shifted * gaussian_weights[offset] * active.to(shifted.dtype)
        alive = active
    return prior


def _get_lengths(one_hot: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is not None:
        return attention_mask.to(one_hot.device).sum(dim=-1).long()
    return one_hot.new_full((one_hot.size(0),), one_hot.size(1), dtype=torch.long)


def _get_padded_length(length: int, min_length: int, multiple: int) -> int:
    if length <= min_length:
        return min_length
    return ((length - 1) // multiple + 1) * multiple


def _fit_length(one_hot: Tensor, length: int) -> Tensor:
    if one_hot.size(1) == length:
        return one_hot
    if one_hot.size(1) > length:
        return one_hot[:, :length]
    return F.pad(one_hot, (0, 0, 0, length - one_hot.size(1)))


def _crop_batch(tensor: Tensor, lengths: Tensor, max_length: int) -> Tensor:
    output = tensor.new_zeros((tensor.size(0), max_length, max_length))
    for batch_index, length in enumerate(lengths.tolist()):
        output[batch_index, :length, :length] = tensor[batch_index, :length, :length]
    return output


def _pair_mask(lengths: Tensor, max_length: int, device: torch.device) -> Tensor:
    positions = torch.arange(max_length, device=device)
    valid = positions.unsqueeze(0) < lengths.to(device).unsqueeze(1)
    return valid.unsqueeze(1) & valid.unsqueeze(2)


def _constraint_matrix(one_hot: Tensor, allow_noncanonical: bool = False) -> Tensor:
    base_a = one_hot[:, :, 0]
    base_c = one_hot[:, :, 1]
    base_g = one_hot[:, :, 2]
    base_u = one_hot[:, :, 3]

    au = base_a.unsqueeze(-1) * base_u.unsqueeze(1)
    cg = base_c.unsqueeze(-1) * base_g.unsqueeze(1)
    ug = base_u.unsqueeze(-1) * base_g.unsqueeze(1)
    mask = au + au.transpose(-1, -2) + cg + cg.transpose(-1, -2) + ug + ug.transpose(-1, -2)

    if allow_noncanonical:
        ac = base_a.unsqueeze(-1) * base_c.unsqueeze(1)
        ag = base_a.unsqueeze(-1) * base_g.unsqueeze(1)
        uc = base_u.unsqueeze(-1) * base_c.unsqueeze(1)
        aa = base_a.unsqueeze(-1) * base_a.unsqueeze(1)
        uu = base_u.unsqueeze(-1) * base_u.unsqueeze(1)
        cc = base_c.unsqueeze(-1) * base_c.unsqueeze(1)
        gg = base_g.unsqueeze(-1) * base_g.unsqueeze(1)
        mask = mask + ac + ac.transpose(-1, -2) + ag + ag.transpose(-1, -2) + uc + uc.transpose(-1, -2)
        mask = mask + aa + uu + cc + gg

    return mask


def _contact_a(a_hat: Tensor, mask: Tensor) -> Tensor:
    contact = a_hat * a_hat
    contact = (contact + contact.transpose(-1, -2)) / 2
    return contact * mask


@dataclass
class UfoldModelOutput(ModelOutput):
    """
    Output type for UFold.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    contact_map: torch.FloatTensor | None = None
    postprocessed_contact_map: torch.FloatTensor | None = None
