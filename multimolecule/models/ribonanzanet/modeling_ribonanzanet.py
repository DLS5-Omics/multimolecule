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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Tuple
from warnings import warn

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
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
    SequencePredictionHead,
    TokenPredictionHead,
)

from ..configuration_utils import HeadConfig
from .configuration_ribonanzanet import RibonanzaNetConfig


class RibonanzaNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RibonanzaNetConfig
    base_model_prefix = "ribonanzanet"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: dict[str, Any] | None = None
    _no_split_modules = ["RibonanzaNetLayer", "RibonanzaNetEmbeddings"]


class RibonanzaNetModel(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetModel, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 256])
        >>> output["pooler_output"].shape
        torch.Size([1, 256])
    """

    def __init__(self, config: RibonanzaNetConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.gradient_checkpointing = False
        self.embeddings = RibonanzaNetEmbeddings(config)
        self.encoder = RibonanzaNetEncoder(config)
        self.pooler = RibonanzaNetPooler(config) if add_pooling_layer else None
        self.fix_attention_mask = config.fix_attention_mask

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @check_model_inputs
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetModelOutputWithPooling:
        if isinstance(input_ids, NestedTensor):
            input_ids, attention_mask = input_ids.tensor, input_ids.mask
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore[union-attr]

        if attention_mask is None:
            attention_mask = (
                input_ids.ne(self.pad_token_id)
                if self.pad_token_id is not None
                else torch.ones(((batch_size, seq_length)), device=device)
            )
        else:
            # Must make a clone here because the attention mask might be reused in other modules
            # and we need to process it to mimic the behavior of the original implementation.
            # See more in https://github.com/Shujun-He/RibonanzaNet/issues/4
            attention_mask = attention_mask.clone()
        attention_mask = attention_mask.to(self.dtype)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        attention_mask = attention_mask.unsqueeze(-1)

        # attention_probs has shape bsz x n_heads x N x N

        embedding_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            extended_attention_mask=extended_attention_mask,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return RibonanzaNetModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], dtype: torch.dtype | None = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-1)
            if not self.fix_attention_mask:
                attention_mask[attention_mask == 0] = -1
            attention_mask = torch.matmul(attention_mask, attention_mask.transpose(1, 2))
        elif attention_mask.shape != 3:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = attention_mask[:, None, :, :]
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        if self.fix_attention_mask:
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


class RibonanzaNetForSequencePrediction(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForSequencePrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForSequencePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.tensor([[1]]))
        >>> output["logits"].shape
        torch.Size([1, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config)
        self.sequence_head = SequencePredictionHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetSequencePredictorOutput:
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.sequence_head(outputs, labels)
        logits, loss = output.logits, output.loss

        return RibonanzaNetSequencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForTokenPrediction(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForTokenPrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForTokenPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        self.token_head = TokenPredictionHead(config)
        self.head_config = self.token_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetTokenPredictorOutput:
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.token_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return RibonanzaNetTokenPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForContactPrediction(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForContactPrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForContactPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels=torch.randint(2, (1, 5, 5)))
        >>> output["logits"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        self.contact_head = ContactPredictionHead(config)
        self.head_config = self.contact_head.config
        self.require_attentions = self.contact_head.require_attentions

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetContactPredictorOutput:
        if self.require_attentions:
            output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
            if output_attentions is False:
                warn("output_attentions must be True since prediction head requires attentions.")
            kwargs["output_attentions"] = True
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        output = self.contact_head(outputs, attention_mask, input_ids, labels)
        logits, loss = output.logits, output.loss

        return RibonanzaNetContactPredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForPreTraining(RibonanzaNetPreTrainedModel):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        # It should have been named as 2a3_head but Python doesn't allow a number at the beginning of a variable name.
        self.a3c_head = TokenPredictionHead(config)
        self.dms_head = TokenPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_2a3: Tensor | None = None,
        labels_dms: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetForPreTrainingOutput:
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output_2a3 = self.a3c_head(outputs, attention_mask, input_ids, labels_2a3)
        logits_2a3, loss_2a3 = output_2a3.logits, output_2a3.loss

        output_dms = self.dms_head(outputs, attention_mask, input_ids, labels_dms)
        logits_dms, loss_dms = output_dms.logits, output_dms.loss

        losses = tuple(l for l in (loss_2a3, loss_dms) if l is not None)  # noqa: E741
        loss = torch.mean(torch.stack(losses)) if losses else None

        return RibonanzaNetForPreTrainingOutput(
            loss=loss,
            logits_2a3=logits_2a3,
            loss_2a3=loss_2a3,
            logits_dms=logits_dms,
            loss_dms=loss_dms,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForSecondaryStructurePrediction(RibonanzaNetForPreTraining):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForSecondaryStructurePrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForSecondaryStructurePrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels_ss=torch.randint(2, (1, 5, 5)))
        >>> output["logits_ss"].shape
        torch.Size([1, 5, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MeanBackward0>)
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        self.ss_head = RibonanzaNetSecondaryStructurePredictionHead(config)
        self.a3c_head = TokenPredictionHead(config)
        self.dms_head = TokenPredictionHead(config)
        self.head_config = self.ss_head.config

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_ss: Tensor | None = None,
        labels_2a3: Tensor | None = None,
        labels_dms: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetForSecondaryStructurePredictorOutput:
        output_pairwise_states = kwargs.get("output_pairwise_states", self.config.output_pairwise_states)
        if not output_pairwise_states:
            warn("output_pairwise_states must be True since prediction head requires pairwise states.")
        kwargs["output_pairwise_states"] = True
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output_ss = self.ss_head(outputs, attention_mask, input_ids, labels_ss)
        logits_ss, loss_ss = output_ss.logits, output_ss.loss

        output_2a3 = self.a3c_head(outputs, attention_mask, input_ids, labels_2a3)
        logits_2a3, loss_2a3 = output_2a3.logits, output_2a3.loss

        output_dms = self.dms_head(outputs, attention_mask, input_ids, labels_dms)
        logits_dms, loss_dms = output_dms.logits, output_dms.loss

        losses = tuple(l for l in (loss_2a3, loss_dms, loss_ss) if l is not None)  # noqa: E741
        loss = torch.mean(torch.stack(losses)) if losses else None

        return RibonanzaNetForSecondaryStructurePredictorOutput(
            loss=loss,
            logits_ss=logits_ss,
            loss_ss=loss_ss,
            logits_2a3=logits_2a3,
            loss_2a3=loss_2a3,
            logits_dms=logits_dms,
            loss_dms=loss_dms,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForDegradationPrediction(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForDegradationPrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForDegradationPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels_reactivity=torch.randn(1, 5))
        >>> output["logits_reactivity"].shape
        torch.Size([1, 5, 1])
        >>> output["loss"]  # doctest:+ELLIPSIS
        tensor(..., grad_fn=<MeanBackward0>)
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        self.reactivity_head = TokenPredictionHead(config)
        self.deg_Mg_pH10_head = TokenPredictionHead(config)
        self.deg_pH10_head = TokenPredictionHead(config)
        self.deg_Mg_50C_head = TokenPredictionHead(config)
        self.deg_50C_head = TokenPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_reactivity: Tensor | None = None,
        labels_deg_Mg_pH10: Tensor | None = None,
        labels_deg_pH10: Tensor | None = None,
        labels_deg_Mg_50C: Tensor | None = None,
        labels_deg_50C: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetForDegradationPredictorOutput:
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output_reactivity = self.reactivity_head(outputs, attention_mask, input_ids, labels_reactivity)
        logits_reactivity, loss_reactivity = output_reactivity.logits, output_reactivity.loss

        output_deg_Mg_pH10 = self.deg_Mg_pH10_head(outputs, attention_mask, input_ids, labels_deg_Mg_pH10)
        logits_deg_Mg_pH10, loss_deg_Mg_pH10 = output_deg_Mg_pH10.logits, output_deg_Mg_pH10.loss

        output_deg_pH10 = self.deg_pH10_head(outputs, attention_mask, input_ids, labels_deg_pH10)
        logits_deg_pH10, loss_deg_pH10 = output_deg_pH10.logits, output_deg_pH10.loss

        output_deg_Mg_50C = self.deg_Mg_50C_head(outputs, attention_mask, input_ids, labels_deg_Mg_50C)
        logits_deg_Mg_50C, loss_deg_Mg_50C = output_deg_Mg_50C.logits, output_deg_Mg_50C.loss

        output_deg_50C = self.deg_50C_head(outputs, attention_mask, input_ids, labels_deg_50C)
        logits_deg_50C, loss_deg_50C = output_deg_50C.logits, output_deg_50C.loss

        losses = tuple(
            l
            for l in (loss_reactivity, loss_deg_Mg_pH10, loss_deg_pH10, loss_deg_Mg_50C, loss_deg_50C)  # noqa: E741
            if l is not None
        )
        loss = torch.mean(torch.stack(losses)) if losses else None

        return RibonanzaNetForDegradationPredictorOutput(
            loss=loss,
            logits_reactivity=logits_reactivity,
            loss_reactivity=loss_reactivity,
            logits_deg_50C=logits_deg_50C,
            loss_deg_50C=loss_deg_50C,
            logits_deg_Mg_50C=logits_deg_Mg_50C,
            loss_deg_Mg_50C=loss_deg_Mg_50C,
            logits_deg_pH10=logits_deg_pH10,
            loss_deg_pH10=loss_deg_pH10,
            logits_deg_Mg_pH10=logits_deg_Mg_pH10,
            loss_deg_Mg_pH10=loss_deg_Mg_pH10,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetForSequenceDropoutPrediction(RibonanzaNetPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import RibonanzaNetConfig, RibonanzaNetForSequenceDropoutPrediction, RnaTokenizer
        >>> config = RibonanzaNetConfig()
        >>> model = RibonanzaNetForSequenceDropoutPrediction(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input, labels_reactivity=torch.randn(1, 5))
        >>> output["logits_2a3"].shape
        torch.Size([1, 1])
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__(config)
        self.ribonanzanet = RibonanzaNetModel(config, add_pooling_layer=False)
        self.a3c_head = RibonanzaNetSequenceDropoutPredictionHead(config)
        self.dms_head = RibonanzaNetSequenceDropoutPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | NestedTensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | NestedTensor | None = None,
        labels_2a3: Tensor | None = None,
        labels_dms: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...] | RibonanzaNetForDegradationPredictorOutput:
        outputs = self.ribonanzanet(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        output_2a3 = self.a3c_head(outputs, attention_mask, input_ids, labels_2a3)
        logits_2a3, loss_2a3 = output_2a3.logits, output_2a3.loss

        output_dms = self.dms_head(outputs, attention_mask, input_ids, labels_dms)
        logits_dms, loss_dms = output_dms.logits, output_dms.loss

        losses = tuple(l for l in (loss_2a3, loss_dms) if l is not None)  # noqa: E741
        loss = torch.mean(torch.stack(losses)) if losses else None

        return RibonanzaNetSequenceDropoutPredictorOutput(
            loss=loss,
            logits_2a3=logits_2a3,
            loss_2a3=loss_2a3,
            logits_dms=logits_dms,
            loss_dms=loss_dms,
            hidden_states=outputs.hidden_states,
            pairwise_states=outputs.pairwise_states,
            attentions=outputs.attentions,
        )


class RibonanzaNetEmbeddings(nn.Module):
    """
    Construct the sequence embeddings.
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        return inputs_embeds


class PairwiseEmbeddings(nn.Module):
    """
    Construct the pairwise embeddings.
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.triangle_proj = RibonanzaNetTriangleProjection(config)
        self.position_embeddings = nn.Linear(17, config.pairwise_size)

    def forward(self, inputs_embeds: torch.FloatTensor) -> Tensor:
        input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        outer_product = self.triangle_proj(inputs_embeds)

        position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)
        bin_values = torch.arange(-8, 9, device=inputs_embeds.device)
        d = position_ids[:, :, None] - position_ids[:, None, :]
        bdy = torch.tensor(8, device=inputs_embeds.device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        position_embeddings = self.position_embeddings(d_onehot)

        embeddings = outer_product + position_embeddings
        return embeddings


class RibonanzaNetTriangleProjection(nn.Module):
    r"""
    Compute the outer product of the hidden states and apply a linear transformation.
    """

    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, config.pairwise_attention_size)
        self.out_proj = nn.Linear(config.pairwise_attention_size**2, config.pairwise_size)

    def forward(self, hidden_states: torch.FloatTensor, pairwise_states: torch.FloatTensor | None = None):
        hidden_states = self.in_proj(hidden_states)
        triangle_states = hidden_states.unsqueeze(1).unsqueeze(-1) * hidden_states.unsqueeze(2).unsqueeze(-2)
        triangle_states = triangle_states.view(*triangle_states.shape[:-2], -1)
        triangle_states = self.out_proj(triangle_states)
        if pairwise_states is not None:
            triangle_states = triangle_states + pairwise_states
        return triangle_states


class RibonanzaNetEncoder(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.config = config
        self.pairwise_embeddings = PairwiseEmbeddings(config)
        layers = [RibonanzaNetLayer(config) for _ in range(config.num_hidden_layers - 1)]
        layers.append(RibonanzaNetLayer(config, kernel_size=1))
        self.layer = nn.ModuleList(layers)
        self.fix_attention_mask = config.fix_attention_mask

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        extended_attention_mask: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RibonanzaNetModelOutput:
        pairwise_states = self.pairwise_embeddings(hidden_states)
        attention_mask_ = attention_mask.clone() if attention_mask is not None else None

        for i, layer_module in enumerate(self.layer):
            # applies attention_mask for convolution
            if attention_mask is not None:
                if not self.fix_attention_mask:
                    attention_mask = (attention_mask_ == 1).float() if i == 0 else attention_mask_.clone()  # type: ignore[union-attr]  # noqa: E501
                hidden_states = hidden_states * attention_mask

            layer_outputs = layer_module(hidden_states, pairwise_states, extended_attention_mask, **kwargs)
            hidden_states, pairwise_states = layer_outputs[:2]

        return RibonanzaNetModelOutput(
            last_hidden_state=hidden_states,
        )


class RibonanzaNetLayer(GradientCheckpointingLayer):
    def __init__(self, config: RibonanzaNetConfig, kernel_size: int = None):  # type: ignore[assignment]
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        kernel_size = kernel_size or getattr(config, "kernel_size", 3)
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv_norm = nn.LayerNorm(config.hidden_size)
        self.pairwise_bias = nn.Sequential(
            nn.LayerNorm(config.pairwise_size), nn.Linear(config.pairwise_size, config.num_attention_heads, bias=False)
        )
        self.attention = RibonanzaNetAttention(config)
        self.intermediate = RibonanzaNetIntermediate(config)
        self.output = RibonanzaNetOutput(config)
        self.pairwise = PairwiseInteraction(config)

    def forward(
        self,
        hidden_states: Tensor,
        pairwise_states: torch.FloatTensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        hidden_states = hidden_states + self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.conv_norm(hidden_states)
        pairwise_bias = self.pairwise_bias(pairwise_states).permute(0, 3, 1, 2)

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            attention_bias=pairwise_bias,
            **kwargs,
        )
        attention_output = self_attention_outputs[0]

        feedforward_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        pairwise_output = self.pairwise(feedforward_output, pairwise_states, attention_mask)

        return (feedforward_output, pairwise_output) + self_attention_outputs[1:]

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RibonanzaNetAttention(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.self = RibonanzaNetSelfAttention(config)
        self.output = RibonanzaNetSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            attention_bias,
            **kwargs,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RibonanzaNetSelfAttention(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
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

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.fix_attention_mask = config.fix_attention_mask
        self.is_causal = False

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        attention_bias: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor, ...]:
        input_shape = hidden_states.shape[:-1]
        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_interface: Callable | None = None
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if attention_interface is not None:
            attn_mask = attention_bias
            if self.fix_attention_mask and attention_mask is not None:
                attn_mask = attention_mask if attn_mask is None else attn_mask + attention_mask
            attn_output, attn_weights = attention_interface(
                self,
                query_layer,
                key_layer,
                value_layer,
                attn_mask,
                dropout=0.0 if not self.training else self.dropout.p,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return attn_output, attn_weights

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore[attr-defined]
        attention_scores = attention_scores * self.scaling

        if attention_bias is not None:
            attention_scores = attention_scores + attention_bias

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RibonanzaNetModel forward() function)
            if self.fix_attention_mask:
                attention_scores = attention_scores + attention_mask
            else:
                attention_scores = attention_scores.masked_fill(attention_mask == -1, -1e-9)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to

        context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)

        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


class RibonanzaNetSelfOutput(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fix_attention_residual = config.fix_attention_residual
        self.layer_norm2 = None
        if not self.fix_attention_residual:
            self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        if not self.fix_attention_residual:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.layer_norm2(hidden_states + input_tensor)  # type: ignore[misc]
        return hidden_states


class RibonanzaNetIntermediate(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
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


class RibonanzaNetOutput(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class PairwiseInteraction(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.triangle_proj = RibonanzaNetTriangleProjection(config)
        self.triangle_mixer_out = RibonanzaNetPairwiseMixer(config, direction="outgoing")
        self.triangle_mixer_in = RibonanzaNetPairwiseMixer(config, direction="ingoing")
        self.use_triangular_attention = config.use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = RibonanzaNetPairwiseAttention(config, axis="row")
            self.triangle_attention_in = RibonanzaNetPairwiseAttention(config, axis="col")
        self.intermediate = RibonanzaNetPairwiseIntermediate(config)
        self.output = RibonanzaNetPairwiseOutput(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pairwise_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
    ):
        pairwise_states = self.triangle_proj(hidden_states, pairwise_states)
        pairwise_states = self.triangle_mixer_out(pairwise_states, attention_mask)
        pairwise_states = self.triangle_mixer_in(pairwise_states, attention_mask)
        if self.use_triangular_attention:
            pairwise_states = self.triangle_attention_out(pairwise_states, attention_mask, output_attentions=False)[0]
            pairwise_states = self.triangle_attention_in(pairwise_states, attention_mask, output_attentions=False)[0]
        feedforward_output = self.feed_forward_chunk(pairwise_states)
        return feedforward_output

    def feed_forward_chunk(self, pairwise_states):
        intermediate_output = self.intermediate(pairwise_states)
        layer_output = self.output(intermediate_output, pairwise_states)
        return layer_output


class RibonanzaNetPairwiseMixer(nn.Module):
    r"""
    Apply triangular update to the pairwise states.
    """

    def __init__(self, config: RibonanzaNetConfig, direction: str = "outgoing"):
        super().__init__()
        if direction not in {"outgoing", "ingoing"}:
            raise ValueError(f"direction must be either outgoing or ingoing, but got {direction}")
        self.direction = direction
        self.in_norm = nn.LayerNorm(config.pairwise_size)
        self.left_proj = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.right_proj = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.left_gate = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.right_gate = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.out_gate = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.out_norm = nn.LayerNorm(config.pairwise_size)
        self.out_proj = nn.Linear(config.pairwise_size, config.pairwise_size)
        self.fix_attention_mask = config.fix_attention_mask
        dropout_dim = -3 if direction == "outgoing" else -2
        if not config.fix_pairwise_dropout:
            dropout_dim = -3
        self.dropout = AxisDropout(config.hidden_dropout, batch_dim=dropout_dim)

    def forward(self, pairwise_states: Tensor, attention_mask: torch.FloatTensor | None = None):

        residual = pairwise_states
        normed_states = self.in_norm(pairwise_states)
        left = self.left_proj(normed_states)
        right = self.right_proj(normed_states)

        if attention_mask is not None:
            if self.fix_attention_mask:
                attention_mask = attention_mask == 0
            attention_mask = attention_mask.transpose(1, 3)
            left = left * attention_mask
            right = right * attention_mask

        left_gate = self.left_gate(normed_states).sigmoid()
        right_gate = self.right_gate(normed_states).sigmoid()
        out_gate = self.out_gate(normed_states).sigmoid()

        left = left * left_gate
        right = right * right_gate

        if self.direction == "outgoing":
            pairwise_states = torch.matmul(left.permute(0, 3, 1, 2), right.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        elif self.direction == "ingoing":
            pairwise_states = torch.matmul(left.permute(0, 3, 2, 1), right.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        pairwise_states = self.out_norm(pairwise_states)
        pairwise_states = pairwise_states * out_gate
        pairwise_states = self.out_proj(pairwise_states)
        pairwise_states = self.dropout(pairwise_states)
        pairwise_states = pairwise_states + residual
        return pairwise_states


class RibonanzaNetPairwiseAttention(nn.Module):
    def __init__(self, config: RibonanzaNetConfig, axis: str = "row"):
        super().__init__()
        self.triangle = RibonanzaNetPairwiseTriangleAttention(config, axis)
        self.output = RibonanzaNetPairwiseTriangleOutput(config, axis)

    def forward(
        self,
        pairwise_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        triangle_outputs = self.triangle(
            pairwise_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(triangle_outputs[0], pairwise_states)
        outputs = (attention_output,) + triangle_outputs[1:]  # add attentions if we output them
        return outputs


class RibonanzaNetPairwiseTriangleAttention(nn.Module):
    def __init__(self, config: RibonanzaNetConfig, axis: str = "row"):
        super().__init__()
        if config.pairwise_size % config.pairwise_num_attention_heads != 0:
            raise ValueError(
                f"The pairwise_size ({config.pairwise_size}) is not a multiple of the number of attention "
                f"heads ({config.pairwise_num_attention_heads})"
            )

        if axis not in {"row", "col"}:
            raise ValueError(f"axis should be either row or col, but got {axis}")

        self.axis = axis
        self.num_attention_heads = config.pairwise_num_attention_heads
        self.attention_head_size = int(config.pairwise_size / config.pairwise_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.layer_norm = nn.LayerNorm(config.pairwise_size)
        self.query = nn.Linear(config.pairwise_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.pairwise_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.pairwise_size, self.all_head_size, bias=False)

        self.pairwise_bias = nn.Linear(config.pairwise_size, self.num_attention_heads, bias=False)
        self.gate = nn.Sequential(nn.Linear(config.pairwise_size, config.pairwise_size), nn.Sigmoid())
        self.fix_attention_mask = config.fix_attention_mask

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_x_shape)

    def forward(
        self,
        pairwise_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool = False,
    ):
        """
        how to do masking
        for row tri attention:
        attention matrix is brijh, where b is batch, r is row, h is head
        so mask should be b()ijh, i.e. take self attention mask and unsqueeze(1,-1)
        add negative inf to matrix before softmax

        for col tri attention
        attention matrix is bijlh, so take self attention mask and unsqueeze(3,-1)

        take attention_mask and spawn pairwise mask, and unsqueeze accordingly
        """
        pairwise_states = self.layer_norm(pairwise_states)

        query_layer = self.transpose_for_scores(self.query(pairwise_states))
        key_layer = self.transpose_for_scores(self.key(pairwise_states))
        value_layer = self.transpose_for_scores(self.value(pairwise_states))
        scale = query_layer.size(-1) ** 0.5

        attention_bias = self.pairwise_bias(pairwise_states)
        attention_bias = attention_bias.unsqueeze(1).permute(0, 4, 1, 2, 3)

        if self.axis == "row":
            query_layer = query_layer.transpose(2, 3)
            key_layer = key_layer.transpose(2, 3)
        else:
            query_layer = query_layer.permute(0, 2, 3, 1, 4)
            key_layer = key_layer.permute(0, 2, 3, 1, 4)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores.transpose(1, 2)
        attention_scores = attention_scores / scale + attention_bias

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RibonanzaNetModel forward() function)
            attention_mask = attention_mask.unsqueeze(1)
            if self.fix_attention_mask:
                attention_scores = attention_scores + attention_mask
            else:
                attention_scores = attention_scores.masked_fill(attention_mask == -1, -1e-9)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if self.axis == "row":
            value_layer = value_layer.transpose(2, 3)
            context_layer = torch.matmul(attention_probs.transpose(1, 2), value_layer)
            context_layer = context_layer.transpose(2, 3)
        else:
            value_layer = value_layer.permute(0, 2, 3, 1, 4)
            context_layer = torch.matmul(attention_probs.transpose(1, 2), value_layer)
            context_layer = context_layer.permute(0, 3, 1, 2, 4)

        context_layer = context_layer.reshape(*context_layer.shape[:-2], -1)
        gate = self.gate(pairwise_states)
        context_layer = gate * context_layer

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class RibonanzaNetPairwiseTriangleOutput(nn.Module):
    def __init__(self, config: RibonanzaNetConfig, axis: str = "row"):
        super().__init__()
        self.dense = nn.Linear(config.pairwise_size, config.pairwise_size)
        dropout_dim = -3 if axis == "row" else -2
        self.dropout = AxisDropout(config.hidden_dropout, batch_dim=dropout_dim)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class RibonanzaNetPairwiseIntermediate(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.pairwise_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.pairwise_size, config.pairwise_intermediate_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.pairwise_hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class RibonanzaNetPairwiseOutput(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
        super().__init__()
        self.dense = nn.Linear(config.pairwise_intermediate_size, config.pairwise_size)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states + input_tensor


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RibonanzaNetPooler(nn.Module):
    def __init__(self, config: RibonanzaNetConfig):
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


class AxisDropout(nn.Dropout):
    """
    Implementation of dropout with the ability to share the dropout mask
    along particular dimensions.

    If not in training mode, this module computes the identity function.

    Examples:
        >>> # Shared dropout along first dimension
        >>> m = AxisDropout(p=0.1, batch_dim=0)
        >>> x = torch.randn(32, 64, 128)
        >>> out = m(x)  # Shape: (32, 64, 128)

        >>> # Shared dropout along multiple dimensions
        >>> m = AxisDropout(p=0.1, batch_dim=[0, 1])
        >>> out = m(x)  # Shape: (32, 64, 128)
    """

    def __init__(self, p: float, batch_dim: int | Sequence[int], inplace: bool = False) -> None:
        """
        Args:
            p: Probability of an element to be zeroed, must be in [0, 1).
            batch_dim: Dimension(s) along which the dropout mask is shared.
            inplace: If set to True, will do this operation in-place.

        Raises:
            ValueError: If dropout probability is invalid.
        """
        super().__init__(p=p, inplace=inplace)

        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.batch_dim = [batch_dim] if isinstance(batch_dim, int) else list(batch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor to which dropout is applied.

        Returns:
            Tensor of the same shape as the input but with dropout applied.

        Raises:
            ValueError: If a dimension in batch_dim is invalid for the input tensor.
        """
        # If we are not in training mode or if p == 0, do nothing.
        if not self.training or self.p == 0:
            return x

        # Validate the requested batch dimensions.
        for dim in self.batch_dim:
            if dim >= x.ndim or dim < -x.ndim:
                raise ValueError(f"Invalid batch_dim {dim} for tensor of rank {x.ndim}")

        # Shape for the broadcasted dropout mask.
        shape = [1 if i in self.batch_dim else s for i, s in enumerate(x.shape)]
        mask = torch.ones(shape, device=x.device, dtype=x.dtype)

        # Use native F.dropout on the mask for consistent scaling and randomness.
        mask = F.dropout(mask, p=self.p, training=self.training, inplace=self.inplace)

        return x * mask


class RibonanzaNetSecondaryStructurePredictionHead(BasePredictionHead):

    output_name: str = "pairwise_states"

    def __init__(self, config: RibonanzaNetConfig, head_config: HeadConfig | None = None):
        if head_config is None:
            head_config = config.head or HeadConfig(num_labels=1)
        super().__init__(config, head_config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.decoder = nn.Linear(config.pairwise_size, self.config.num_labels)
        self.criterion = Criterion(self.config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: RibonanzaNetModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            pairwise_state = outputs["pairwise_states"][-1]
        elif isinstance(outputs, tuple):
            if outputs[2][0].ndim == 4:
                pairwise_state = outputs[2][-1]
            elif outputs[3][0].ndim == 4:
                pairwise_state = outputs[3][-1]
            else:
                raise ValueError(
                    "Could not find pairwise states in model outputs. This is likely a bug - please report it at "
                    "https://github.com/multimolecule/multimolecule/issues. As a temporary workaround, set "
                    "return_dict=True when calling the model."
                )
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        pairwise_state, _, _ = self.remove_special_tokens_2d(pairwise_state, attention_mask, input_ids)

        pairwise_state = pairwise_state + pairwise_state.transpose(1, 2)

        output = self.dropout(pairwise_state)
        output = self.decoder(output)

        if labels is not None:
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)


class RibonanzaNetSequenceDropoutPredictionHead(BasePredictionHead):
    def __init__(self, config: RibonanzaNetConfig, head_config: HeadConfig | None = None):
        if head_config is None:
            head_config = config.head or HeadConfig(num_labels=1)
        super().__init__(config, head_config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.decoder = nn.Linear(config.hidden_size, self.config.num_labels)
        self.criterion = Criterion(self.config)

    def forward(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        outputs: RibonanzaNetModelOutput | Mapping[str, Tensor] | Tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: NestedTensor | Tensor | None = None,
        labels: Tensor | None = None,
    ) -> HeadOutput:
        if isinstance(outputs, (Mapping, ModelOutput)):
            output = outputs["last_hidden_state"]
        elif isinstance(outputs, tuple):
            output = outputs[0]
        else:
            raise ValueError(f"Unsupported type for outputs: {type(outputs)}")

        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, _, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        output = self.dropout(output)
        output = self.decoder(output).mean(dim=1).exp()

        if labels is not None:
            return HeadOutput(output, self.criterion(output, labels))
        return HeadOutput(output)


@dataclass
class RibonanzaNetModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetSequencePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetTokenPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetContactPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_2a3: torch.FloatTensor = None
    loss_2a3: torch.FloatTensor | None = None
    logits_dms: torch.FloatTensor = None
    loss_dms: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetForSecondaryStructurePredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_ss: torch.FloatTensor = None
    loss_ss: torch.FloatTensor | None = None
    logits_2a3: torch.FloatTensor = None
    loss_2a3: torch.FloatTensor | None = None
    logits_dms: torch.FloatTensor = None
    loss_dms: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetForDegradationPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_reactivity: torch.FloatTensor = None
    loss_reactivity: torch.FloatTensor | None = None
    logits_deg_Mg_pH10: torch.FloatTensor = None
    loss_deg_Mg_pH10: torch.FloatTensor | None = None
    logits_deg_pH10: torch.FloatTensor = None
    loss_deg_pH10: torch.FloatTensor | None = None
    logits_deg_Mg_50C: torch.FloatTensor = None
    loss_deg_Mg_50C: torch.FloatTensor | None = None
    logits_deg_50C: torch.FloatTensor = None
    loss_deg_50C: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RibonanzaNetSequenceDropoutPredictorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_2a3: torch.FloatTensor = None
    loss_2a3: torch.FloatTensor | None = None
    logits_dms: torch.FloatTensor = None
    loss_dms: torch.FloatTensor | None = None
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    pairwise_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


RibonanzaNetPreTrainedModel._can_record_outputs = {
    "hidden_states": RibonanzaNetLayer,
    "pairwise_states": [
        OutputRecorder(PairwiseEmbeddings, index=0),
        OutputRecorder(RibonanzaNetLayer, index=1),
    ],
    "attentions": OutputRecorder(RibonanzaNetAttention, index=1, layer_name="attention"),
}
