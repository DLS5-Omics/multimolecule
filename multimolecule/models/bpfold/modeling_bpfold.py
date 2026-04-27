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
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from multimolecule.modules import preserve_batch_norm_stats

from .configuration_bpfold import BpfoldConfig


class BpfoldPreTrainedModel(PreTrainedModel):

    config_class = BpfoldConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["BpfoldLayer"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)


class BpfoldModel(BpfoldPreTrainedModel):
    outer_energy: Tensor
    inner_chain_energy: Tensor
    inner_hairpin_energy: Tensor
    _pair_index: Tensor

    """
    Examples:
        >>> import torch
        >>> from multimolecule import BpfoldConfig, BpfoldModel
        >>> config = BpfoldConfig(
        ...     hidden_size=8, attention_head_size=4, intermediate_size=16, num_hidden_layers=1, num_members=1
        ... )
        >>> model = BpfoldModel(config)
        >>> input_ids = torch.tensor([[1, 6, 7, 8, 9, 2]])
        >>> output = model(input_ids=input_ids)
        >>> output["logits"].shape
        torch.Size([1, 4, 4])
        >>> output["contact_map"].shape
        torch.Size([1, 4, 4])
    """

    def __init__(self, config: BpfoldConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.members = nn.ModuleList([BpfoldModule(config) for _ in range(config.num_members)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight]))
        self.supports_batch_process = True

        outer_shape, inner_chain_shape, inner_hairpin_shape = _energy_table_shapes(
            num_bases=4,
            motif_radius=config.motif_radius,
        )
        self.register_buffer("outer_energy", torch.zeros(outer_shape))
        self.register_buffer("inner_chain_energy", torch.zeros(inner_chain_shape))
        self.register_buffer("inner_hairpin_energy", torch.zeros(inner_hairpin_shape))
        self.register_buffer("_pair_index", _pair_index_matrix(), persistent=False)

        self.post_init()

    def postprocess(self, outputs, input_ids=None, **kwargs):
        postprocessed_contact_map = outputs.get("postprocessed_contact_map")
        if postprocessed_contact_map is not None:
            return postprocessed_contact_map
        return outputs["contact_map"]

    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        base_pair_energy: Tensor | None = None,
        base_pair_probability: Tensor | None = None,
        use_postprocessing: bool | None = None,
        return_noncanonical: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BpfoldModelOutput:
        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is not None and isinstance(inputs_embeds, NestedTensor):
            raise TypeError("BpfoldModel does not support NestedTensor inputs_embeds")

        if inputs_embeds is not None:
            inputs_embeds, attention_mask = self._prepare_inputs_embeds(inputs_embeds, attention_mask)
            network_length = inputs_embeds.size(1)
            base_one_hot = None
            valid_mask = _pair_mask(attention_mask.sum(dim=-1).long(), network_length, attention_mask.device)
            if self.config.use_base_pair_energy:
                if base_pair_energy is None:
                    raise ValueError(
                        "base_pair_energy must be provided when using inputs_embeds with use_base_pair_energy=True."
                    )
                base_pair_energy = _fit_pairwise_feature(base_pair_energy, network_length)
            else:
                base_pair_energy = None
            if base_pair_probability is not None:
                base_pair_probability = _fit_pairwise_feature(base_pair_probability, network_length)
            token_ids = None
        else:
            token_ids, attention_mask, base_indices, base_lengths, base_one_hot = self._prepare_input_ids(
                input_ids, attention_mask
            )
            network_length = token_ids.size(1)
            base_start = 1 if self.config.bos_token_id is not None else 0
            if self.config.use_base_pair_energy:
                if base_pair_energy is None:
                    base_pair_energy = self._base_pair_energy(base_indices, base_lengths, network_length, base_start)
                else:
                    base_pair_energy = _pad_pairwise_feature(base_pair_energy, base_lengths, network_length, base_start)
            else:
                base_pair_energy = None
            if base_pair_probability is not None:
                base_pair_probability = _pad_pairwise_feature(
                    base_pair_probability, base_lengths, network_length, base_start
                )

        member_logits = [
            member(
                input_ids=token_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                base_pair_energy=base_pair_energy,
                base_pair_probability=base_pair_probability,
            )
            for member in self.members
        ]
        logits_with_tokens = torch.stack(member_logits, dim=0).mean(dim=0)

        if inputs_embeds is not None:
            logits = logits_with_tokens
        else:
            logits, valid_mask, _ = self._remove_special_tokens_2d(
                logits_with_tokens.unsqueeze(-1), attention_mask, token_ids
            )
            logits = logits.squeeze(-1)
            valid_mask = valid_mask.bool()

        should_postprocess = self.config.use_postprocessing if use_postprocessing is None else use_postprocessing
        postprocessed_contact_map = None
        noncanonical_contact_map = None
        if should_postprocess:
            if base_one_hot is None:
                raise ValueError("input_ids are required for BPfold post-processing when using inputs_embeds.")
            with torch.no_grad():
                postprocessed_contact_map = self._postprocess(logits, base_one_hot, is_noncanonical=False)
                if return_noncanonical:
                    noncanonical_contact_map = self._postprocess(logits, base_one_hot, is_noncanonical=True)
            contact_map = postprocessed_contact_map
        else:
            contact_map = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            labels = labels.to(device=logits.device, dtype=logits.dtype)
            max_length = logits.size(-1)
            labels = labels[:, :max_length, :max_length]
            loss = self.criterion(logits[valid_mask], labels[valid_mask])

        return BpfoldModelOutput(
            loss=loss,
            logits=logits,
            contact_map=contact_map,
            postprocessed_contact_map=postprocessed_contact_map,
            noncanonical_contact_map=noncanonical_contact_map,
        )

    def _prepare_inputs_embeds(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if inputs_embeds.size(-1) != self.config.hidden_size:
            raise ValueError(
                f"inputs_embeds last dimension ({inputs_embeds.size(-1)}) must equal hidden_size "
                f"({self.config.hidden_size})."
            )
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.bool, device=inputs_embeds.device)
        else:
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        network_length = int(attention_mask.long().sum(dim=-1).max().item()) if attention_mask.numel() > 0 else 0
        return inputs_embeds[:, :network_length], attention_mask[:, :network_length]

    def _prepare_input_ids(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        network_length = int(attention_mask.long().sum(dim=-1).max().item()) if attention_mask.numel() > 0 else 0
        token_ids = input_ids[:, :network_length].clamp(min=0, max=self.config.vocab_size - 1)
        attention_mask = attention_mask[:, :network_length]
        _, base_mask, base_token_ids = self._remove_special_tokens(
            token_ids.new_ones((*token_ids.shape, 1), dtype=torch.float32),
            attention_mask,
            token_ids,
        )
        base_mask = base_mask.bool()
        base_lengths = base_mask.sum(dim=-1).long()
        base_indices = self._base_indices(base_token_ids)
        base_one_hot = F.one_hot(base_indices, num_classes=4).float()
        base_one_hot = base_one_hot * base_mask.unsqueeze(-1).to(base_one_hot.dtype)
        return token_ids, attention_mask, base_indices, base_lengths, base_one_hot

    def _remove_special_tokens(
        self,
        output: Tensor,
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.bos_token_id is not None:
            output = output[..., 1:, :]
            if attention_mask is not None:
                attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        if self.config.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.config.eos_token_id).to(output.device)
                input_ids = input_ids.masked_fill(~eos_mask, self.config.pad_token_id or 0)[..., :-1]
            elif attention_mask is not None:
                last_valid_indices = attention_mask.sum(dim=-1) - 1
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) != last_valid_indices.unsqueeze(1)
            else:
                raise ValueError("Unable to remove EOS tokens because input_ids and attention_mask are both None")
            output = (output * eos_mask.unsqueeze(-1))[..., :-1, :]
            if attention_mask is not None:
                attention_mask = (attention_mask * eos_mask)[..., :-1]
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output, attention_mask, input_ids

    def _remove_special_tokens_2d(
        self,
        output: Tensor,
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.bos_token_id is not None:
            output = output[..., 1:, 1:, :]
            if attention_mask is not None:
                attention_mask = attention_mask[..., 1:]
            if input_ids is not None:
                input_ids = input_ids[..., 1:]
        if self.config.eos_token_id is not None:
            if input_ids is not None:
                eos_mask = input_ids.ne(self.config.eos_token_id).to(output.device)
                input_ids = input_ids.masked_fill(~eos_mask, self.config.pad_token_id or 0)[..., :-1]
            elif attention_mask is not None:
                last_valid_indices = attention_mask.sum(dim=-1) - 1
                seq_length = attention_mask.size(-1)
                eos_mask = torch.arange(seq_length, device=output.device) != last_valid_indices.unsqueeze(1)
            else:
                raise ValueError("Unable to remove EOS tokens because input_ids and attention_mask are both None")
            if attention_mask is not None:
                attention_mask = (attention_mask * eos_mask)[..., :-1]
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            output = (output * eos_mask.unsqueeze(-1))[..., :-1, :-1, :]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            output = output * attention_mask.unsqueeze(-1)
        return output, attention_mask, input_ids

    def _base_indices(self, input_ids: Tensor) -> Tensor:
        if self.config.null_token_id is None:
            raise ValueError("BpfoldModel requires null_token_id to infer nucleotide token ids from input_ids.")
        base_offset = self.config.null_token_id + 1
        base_indices = input_ids - base_offset
        return torch.where(
            (base_indices >= 0) & (base_indices < 4),
            base_indices,
            torch.full_like(base_indices, 3),
        )

    def _base_pair_energy(
        self,
        base_indices: Tensor,
        base_lengths: Tensor,
        target_length: int,
        base_start: int,
    ) -> Tensor:
        batch_size = base_indices.size(0)
        num_channels = 2 if self.config.separate_outer_inner_energy else 1
        energy = base_indices.new_zeros((batch_size, num_channels, target_length, target_length), dtype=torch.float32)

        for batch_index in range(batch_size):
            length = int(base_lengths[batch_index].item())
            if length <= 0:
                continue
            seq = base_indices[batch_index, :length]
            seq_energy = _build_energy_map_from_tokens(
                seq,
                self._pair_index,
                self.outer_energy,
                self.inner_chain_energy,
                self.inner_hairpin_energy,
                num_bases=4,
                motif_radius=self.config.motif_radius,
                separate_outer_inner=self.config.separate_outer_inner_energy,
            )
            base_end = base_start + length
            energy[batch_index, :, base_start:base_end, base_start:base_end] = seq_energy
        return energy

    def _postprocess(self, logits: Tensor, base_one_hot: Tensor, is_noncanonical: bool = False) -> Tensor:
        if is_noncanonical:
            rho = self.config.postprocess_nc_rho
            threshold_logit = self.config.postprocess_nc_s
        else:
            rho = self.config.postprocess_rho
            threshold_logit = self.config.postprocess_s

        mask = _constraint_matrix(base_one_hot, is_noncanonical=is_noncanonical).float()
        u = torch.sigmoid(2 * (logits - threshold_logit)) * logits
        a_hat = torch.sigmoid(u) * torch.sigmoid(2 * (u - threshold_logit)).detach()
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
                a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

            lmbd_grad = F.relu(_contact_a(a_hat, mask).sum(dim=-1) - 1)
            lmbd = lmbd + lr_max * lmbd_grad
            lr_max *= 0.99

        contact_map = _contact_a(a_hat, mask)
        contact_map = (contact_map > self.config.threshold).float()
        if is_noncanonical:
            contact_map = contact_map * _noncanonical_matrix(base_one_hot).float()
        return contact_map


class BpfoldModule(nn.Module):
    def __init__(self, config: BpfoldConfig):
        super().__init__()
        self.use_base_pair_energy = config.use_base_pair_energy
        self.use_base_pair_probability = config.use_base_pair_probability
        self.separate_outer_inner_energy = config.separate_outer_inner_energy
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        pairwise_channels = 0
        if config.use_base_pair_probability:
            pairwise_channels += 1
        if config.use_base_pair_energy:
            pairwise_channels += 2 if config.separate_outer_inner_energy else 1

        self.encoder = BpfoldEncoder(
            hidden_size=config.hidden_size,
            attention_head_size=config.attention_head_size,
            intermediate_size=config.intermediate_size,
            hidden_dropout=config.hidden_dropout,
            positional_embedding=config.positional_embedding,
            num_hidden_layers=config.num_hidden_layers,
            num_pairwise_convolutions=config.num_pairwise_convolutions,
            pairwise_kernel_size=config.pairwise_kernel_size,
            use_squeeze_excitation=config.use_squeeze_excitation,
            pairwise_channels=pairwise_channels,
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        base_pair_energy: Tensor | None = None,
        base_pair_probability: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is None:
            raise ValueError("attention_mask must be provided.")
        sequence_length = int(attention_mask.sum(dim=-1).max().item())
        attention_mask = attention_mask[:, :sequence_length]
        if inputs_embeds is not None:
            hidden_states = inputs_embeds[:, :sequence_length]
        else:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            input_ids = input_ids[:, :sequence_length]
            hidden_states = self.embeddings(input_ids)

        pairwise_features = []
        if self.use_base_pair_probability:
            if base_pair_probability is None:
                base_pair_probability = hidden_states.new_zeros(
                    hidden_states.size(0), 1, sequence_length, sequence_length
                )
            pairwise_features.append(base_pair_probability[:, :, :sequence_length, :sequence_length])
        if self.use_base_pair_energy:
            if base_pair_energy is None:
                channels = 2 if self.separate_outer_inner_energy else 1
                base_pair_energy = hidden_states.new_zeros(
                    hidden_states.size(0), channels, sequence_length, sequence_length
                )
            pairwise_features.append(base_pair_energy[:, :, :sequence_length, :sequence_length])
        attention_bias = torch.cat(pairwise_features, dim=1) if pairwise_features else None

        hidden_states = self.encoder(hidden_states, attention_bias, attention_mask=attention_mask)
        return hidden_states @ hidden_states.transpose(-1, -2)


class BpfoldEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_head_size: int,
        intermediate_size: int,
        hidden_dropout: float,
        positional_embedding: str,
        num_hidden_layers: int,
        num_pairwise_convolutions: int,
        pairwise_kernel_size: int,
        use_squeeze_excitation: bool,
        pairwise_channels: int,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        num_heads = hidden_size // attention_head_size
        self.layer = nn.ModuleList(
            [
                BpfoldLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    positional_embedding=positional_embedding,
                    dropout=hidden_dropout,
                    intermediate_size=intermediate_size,
                    use_squeeze_excitation=use_squeeze_excitation,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.pairwise_channels = pairwise_channels
        if pairwise_channels > 0:
            self.pairwise_blocks = nn.ModuleList(
                [
                    BpfoldPairwiseBlock(
                        pairwise_channels if layer_index == 0 else num_heads,
                        num_heads,
                        kernel_size=pairwise_kernel_size,
                        use_squeeze_excitation=use_squeeze_excitation,
                    )
                    for layer_index in range(num_pairwise_convolutions)
                ]
            )
        else:
            self.pairwise_blocks = nn.ModuleList()

    def forward(
        self,
        hidden_states: Tensor,
        attention_bias: Tensor | None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        for layer_index, layer_module in enumerate(self.layer):
            if attention_bias is not None and layer_index < len(self.pairwise_blocks):
                pairwise_block = self.pairwise_blocks[layer_index]
                if self.gradient_checkpointing and self.training:
                    attention_bias = self._gradient_checkpointing_func(
                        pairwise_block.__call__,
                        attention_bias,
                        context_fn=lambda pairwise_block=pairwise_block: (
                            nullcontext(),
                            preserve_batch_norm_stats(pairwise_block),
                        ),
                    )
                else:
                    attention_bias = pairwise_block(attention_bias)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_bias,
                    attention_mask,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_bias=attention_bias,
                    attention_mask=attention_mask,
                )
        return hidden_states


class BpfoldLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        positional_embedding: str,
        num_heads: int,
        dropout: float,
        intermediate_size: int,
        use_squeeze_excitation: bool,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = BpfoldSelfAttention(
            hidden_size=hidden_size,
            positional_embedding=positional_embedding,
            num_heads=num_heads,
            dropout=dropout,
            use_squeeze_excitation=use_squeeze_excitation,
        )
        self.dropout = nn.Dropout(dropout)
        self.intermediate = BpfoldIntermediate(hidden_size, intermediate_size, dropout)
        self.output = BpfoldOutput(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_bias: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attention_output = self.attention(
            self.layer_norm(hidden_states),
            attention_bias=attention_bias,
            attention_mask=attention_mask,
        )
        attention_output = self.dropout(attention_output) + hidden_states
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class BpfoldSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        positional_embedding: str,
        num_heads: int,
        dropout: float,
        temperature: float = 1.0,
        use_squeeze_excitation: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.temperature = temperature
        self.use_squeeze_excitation = use_squeeze_excitation
        self.dynamic_position_bias: BpfoldDynamicPositionBias | None
        self.alibi_position_bias: BpfoldAlibiPositionBias | None

        if positional_embedding == "dyn":
            self.dynamic_position_bias = BpfoldDynamicPositionBias(
                hidden_size=hidden_size // 4, num_heads=num_heads, depth=2
            )
            self.alibi_position_bias = None
        else:
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi_position_bias = BpfoldAlibiPositionBias(alibi_heads, num_heads)
            self.dynamic_position_bias = None

        self.dropout = nn.Dropout(dropout)
        self.query_key_value_weight = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
        self.output_weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.output_bias = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.query_key_value_bias = nn.Parameter(torch.empty(1, 1, 3 * hidden_size))
        if not use_squeeze_excitation:
            self.gamma = nn.Parameter(torch.ones(num_heads).view(1, -1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.query_key_value_weight)
        nn.init.xavier_normal_(self.output_weight)
        nn.init.zeros_(self.output_bias)
        nn.init.zeros_(self.query_key_value_bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_bias: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = hidden_states @ self.query_key_value_weight + self.query_key_value_bias
        query, key, value = (
            qkv.view(batch_size, sequence_length, self.num_heads, -1).permute(0, 2, 1, 3).chunk(3, dim=-1)
        )
        attention = query @ key.transpose(-1, -2)
        attention = attention / self.temperature / math.sqrt(self.head_size)
        if self.dynamic_position_bias is not None:
            attention = attention + self.dynamic_position_bias(sequence_length, sequence_length).unsqueeze(0)
        if self.alibi_position_bias is not None:
            attention = attention + self.alibi_position_bias(sequence_length, sequence_length).unsqueeze(0)
        if attention_bias is not None:
            if not self.use_squeeze_excitation:
                attention_bias = self.gamma * attention_bias
            attention = attention + attention_bias
        if attention_mask is not None:
            attention = attention.masked_fill(~attention_mask[:, None, None, :], torch.finfo(attention.dtype).min)
        attention = attention.softmax(dim=-1)
        if attention_mask is not None:
            attention = attention * attention_mask[:, None, :, None].to(attention.dtype)
        output = attention @ value
        output = output.permute(0, 2, 1, 3).flatten(2, 3)
        return output + self.output_bias


class BpfoldIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.dropout(hidden_states)


class BpfoldOutput(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class BpfoldPairwiseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_squeeze_excitation: bool,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.squeeze_excitation = BpfoldSqueezeExcitation(out_channels) if use_squeeze_excitation else None
        self.activation = nn.GELU()
        self.residual: nn.Module
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.conv(hidden_states)
        output = self.batch_norm(output)
        if self.squeeze_excitation is not None:
            output = self.squeeze_excitation(output)
        output = self.activation(output)
        return output + self.residual(hidden_states)


class BpfoldSqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(channels, channels // reduction, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(channels // reduction, channels, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, channels, _, _ = hidden_states.shape
        scale = self.squeeze(hidden_states).view(batch_size, channels)
        scale = self.dense1(scale)
        scale = self.activation(scale)
        scale = self.dense2(scale)
        scale = self.gate(scale).view(batch_size, channels, 1, 1)
        return hidden_states * scale.expand_as(hidden_states)


class BpfoldDynamicPositionBias(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        log_distance: bool = False,
        norm: bool = False,
    ):
        super().__init__()
        self.log_distance = log_distance
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                    nn.SiLU(),
                )
            ]
        )
        for _ in range(depth - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                    nn.SiLU(),
                )
            )
        self.mlp.append(nn.Linear(hidden_size, num_heads))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, query_length: int, key_length: int) -> Tensor:
        if query_length != key_length:
            raise ValueError("BpfoldDynamicPositionBias requires equal query and key lengths.")
        positions = torch.arange(query_length, device=self.device)
        relative_indices = positions[:, None] - positions[None, :] + query_length - 1
        relative_positions = torch.arange(-query_length + 1, query_length, device=self.device, dtype=torch.float32)
        relative_positions = relative_positions[:, None]
        if self.log_distance:
            relative_positions = torch.sign(relative_positions) * torch.log(relative_positions.abs() + 1)
        for layer in self.mlp:
            relative_positions = layer(relative_positions)
        bias = relative_positions[relative_indices]
        return bias.permute(2, 0, 1)


class BpfoldAlibiPositionBias(nn.Module):
    slopes: Tensor
    bias: Tensor | None

    def __init__(self, num_heads: int, total_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.total_heads = total_heads
        slopes = torch.tensor(self._get_slopes(num_heads), dtype=torch.float32).view(num_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.slopes.device

    def forward(self, query_length: int, key_length: int) -> Tensor:
        if self.bias is not None and self.bias.shape[-2] >= query_length and self.bias.shape[-1] >= key_length:
            return self.bias[..., -query_length:, -key_length:]
        query_positions = torch.arange(key_length - query_length, key_length, device=self.device)
        key_positions = torch.arange(key_length, device=self.device)
        bias = -torch.abs(key_positions[None, None, :] - query_positions[None, :, None])
        bias = bias * self.slopes
        if self.total_heads > self.num_heads:
            padding = bias.new_zeros((self.total_heads - self.num_heads, query_length, key_length))
            bias = torch.cat([bias, padding], dim=0)
        self.register_buffer("bias", bias, persistent=False)
        return bias

    @staticmethod
    def _get_slopes(num_heads: int) -> list[float]:
        def get_power_of_2_slopes(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_power_of_2_slopes(num_heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        extra = get_power_of_2_slopes(2 * closest_power_of_2)[0::2]
        return get_power_of_2_slopes(closest_power_of_2) + extra[: num_heads - closest_power_of_2]


def _build_energy_map_from_tokens(
    seq: Tensor,
    pair_index: Tensor,
    outer_energy: Tensor,
    inner_chain_energy: Tensor,
    inner_hairpin_energy: Tensor,
    num_bases: int,
    motif_radius: int,
    separate_outer_inner: bool,
) -> Tensor:
    length = seq.size(0)
    channels = 2 if separate_outer_inner else 1
    energy = seq.new_zeros((channels, length, length), dtype=torch.float32)
    min_pair_distance = motif_radius + 1
    if length < min_pair_distance + 1:
        return energy

    i, j = torch.triu_indices(length, length, offset=min_pair_distance, device=seq.device)
    pair = pair_index[seq[i], seq[j]]
    valid = pair.ge(0)
    i, j, pair = i[valid], j[valid], pair[valid]

    right = torch.minimum(j + motif_radius, j.new_full((), length - 1))
    left = torch.maximum(i - motif_radius, i.new_zeros(()))
    outer_values = outer_energy[
        pair_index[seq[j], seq[i]],
        right - j,
        i - left,
        _encode_tensor_ranges(seq, j + 1, right + 1, max_width=motif_radius, num_bases=num_bases),
        _encode_tensor_ranges(seq, left, i, max_width=motif_radius, num_bases=num_bases),
    ]

    distance = j - i
    hairpin_distance = 2 * motif_radius
    hairpin_code_width = hairpin_distance - 1
    chain_left_code = _encode_tensor_ranges(
        seq,
        i + 1,
        i + motif_radius + 1,
        max_width=motif_radius,
        num_bases=num_bases,
    )
    chain_right_code = _encode_tensor_ranges(
        seq,
        j - motif_radius,
        j,
        max_width=motif_radius,
        num_bases=num_bases,
    )
    chain_code = chain_left_code * (num_bases**motif_radius) + chain_right_code
    inner_values = torch.where(
        distance <= hairpin_distance,
        inner_hairpin_energy[
            pair,
            distance.clamp(max=hairpin_distance),
            _encode_tensor_ranges(seq, i + 1, j, max_width=hairpin_code_width, num_bases=num_bases),
        ],
        inner_chain_energy[pair, chain_code],
    )

    if separate_outer_inner:
        energy[0, i, j] = outer_values
        energy[0, j, i] = outer_values
        energy[1, i, j] = inner_values
        energy[1, j, i] = inner_values
    else:
        values = outer_values + inner_values
        energy[0, i, j] = values
        energy[0, j, i] = values
    return energy


def _energy_table_shapes(
    num_bases: int,
    motif_radius: int,
) -> tuple[tuple[int, int, int, int, int], tuple[int, int], tuple[int, int, int]]:
    num_pair_types = 6
    motif_code_size = num_bases**motif_radius
    chain_code_size = num_bases ** (2 * motif_radius)
    max_hairpin_distance = 2 * motif_radius
    return (
        (num_pair_types, motif_radius + 1, motif_radius + 1, motif_code_size, motif_code_size),
        (num_pair_types, chain_code_size),
        (num_pair_types, max_hairpin_distance + 1, chain_code_size),
    )


def _pair_index_matrix() -> Tensor:
    matrix = torch.full((4, 4), -1, dtype=torch.long)
    matrix[2, 1] = 0
    matrix[1, 2] = 1
    matrix[0, 3] = 2
    matrix[3, 0] = 3
    matrix[2, 3] = 4
    matrix[3, 2] = 5
    return matrix


def _encode_tensor_ranges(seq: Tensor, start: Tensor, end: Tensor, max_width: int, num_bases: int) -> Tensor:
    lengths = (end - start).clamp(min=0, max=max_width)
    code = torch.zeros_like(start, dtype=torch.long)
    for offset in range(max_width):
        valid = offset < lengths
        index = (start + offset).clamp(min=0, max=seq.size(0) - 1)
        code = torch.where(valid, code * num_bases + seq[index].long(), code)
    return code


def _sequence_mask(lengths: Tensor, max_length: int, device: torch.device) -> Tensor:
    positions = torch.arange(max_length, device=device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _fit_pairwise_feature(feature: Tensor, target_length: int) -> Tensor:
    if feature.dim() == 3:
        feature = feature.unsqueeze(1)
    if feature.dim() != 4:
        raise ValueError(f"Pairwise feature must have shape (B, L, L) or (B, C, L, L), got {feature.shape}.")
    if feature.size(-1) < target_length or feature.size(-2) < target_length:
        raise ValueError(
            f"Pairwise feature shape {tuple(feature.shape[-2:])} must cover target length {target_length}."
        )
    return feature[:, :, :target_length, :target_length]


def _pad_pairwise_feature(feature: Tensor, lengths: Tensor, target_length: int, base_start: int) -> Tensor:
    if feature.dim() == 3:
        feature = feature.unsqueeze(1)
    if feature.dim() != 4:
        raise ValueError(f"Pairwise feature must have shape (B, L, L) or (B, C, L, L), got {feature.shape}.")
    if feature.size(-1) == target_length and feature.size(-2) == target_length:
        return feature[:, :, :target_length, :target_length]
    output = feature.new_zeros((feature.size(0), feature.size(1), target_length, target_length))
    for batch_index, length in enumerate(lengths.tolist()):
        if length > 0:
            base_end = base_start + length
            output[batch_index, :, base_start:base_end, base_start:base_end] = feature[batch_index, :, :length, :length]
    return output


def _pair_mask(lengths: Tensor, max_length: int, device: torch.device) -> Tensor:
    mask = _sequence_mask(lengths, max_length, device)
    return mask.unsqueeze(1) & mask.unsqueeze(2)


def _constraint_matrix(x: Tensor, loop_min_len: int = 2, is_noncanonical: bool = False) -> Tensor:
    base_a = x[:, :, 0]
    base_c = x[:, :, 1]
    base_g = x[:, :, 2]
    base_u = x[:, :, 3]
    au = base_a.unsqueeze(2) * base_u.unsqueeze(1)
    cg = base_c.unsqueeze(2) * base_g.unsqueeze(1)
    ug = base_u.unsqueeze(2) * base_g.unsqueeze(1)
    pair_mask = au + au.transpose(-1, -2) + cg + cg.transpose(-1, -2) + ug + ug.transpose(-1, -2)
    if is_noncanonical:
        pair_mask = torch.ones_like(pair_mask) - torch.eye(pair_mask.size(-1), device=pair_mask.device).unsqueeze(0)

    length = pair_mask.size(-1)
    sharp_loop = torch.ones(length, length, device=pair_mask.device, dtype=torch.bool).triu(diagonal=loop_min_len + 1)
    sharp_loop = sharp_loop | sharp_loop.transpose(-1, -2)
    return pair_mask * sharp_loop.to(dtype=pair_mask.dtype)


def _noncanonical_matrix(x: Tensor) -> Tensor:
    canonical = _constraint_matrix(x, loop_min_len=2, is_noncanonical=False).bool()
    valid = x.sum(dim=-1) > 0
    all_pairs = (valid.unsqueeze(1) & valid.unsqueeze(2)).bool()
    diagonal = torch.eye(x.size(1), device=x.device, dtype=torch.bool).unsqueeze(0)
    return all_pairs & ~canonical & ~diagonal


def _contact_a(a_hat: Tensor, mask: Tensor) -> Tensor:
    contact = a_hat * a_hat
    contact = (contact + contact.transpose(-1, -2)) / 2
    return contact * mask


@dataclass
class BpfoldModelOutput(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None
    contact_map: Tensor | None = None
    postprocessed_contact_map: Tensor | None = None
    noncanonical_contact_map: Tensor | None = None
