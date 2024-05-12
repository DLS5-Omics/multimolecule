# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import torch
import transformers
import transformers.models
from torch import Tensor
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from multimolecule.models.rnabert import RnaBertConfig, RnaBertModel, RnaBertPreTrainedModel
from multimolecule.models.rnafm import RnaFmConfig, RnaFmModel, RnaFmPreTrainedModel
from multimolecule.models.rnamsm import RnaMsmConfig, RnaMsmModel, RnaMsmPreTrainedModel
from multimolecule.models.splicebert import SpliceBertConfig, SpliceBertModel, SpliceBertPreTrainedModel
from multimolecule.models.utrbert import UtrBertConfig, UtrBertModel, UtrBertPreTrainedModel
from multimolecule.models.utrlm import UtrLmConfig, UtrLmModel, UtrLmPreTrainedModel
from multimolecule.module import ClassificationHead


class AutoModelForCrisprOffTarget(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())


transformers.models.auto.modeling_auto.AutoModelForCrisprOffTarget = AutoModelForCrisprOffTarget
transformers.AutoModelForCrisprOffTarget = AutoModelForCrisprOffTarget


class RnaBertForCrisprOffTarget(RnaBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaBertConfig, RnaBertForCrisprOffTarget, RnaTokenizer
        >>> config = RnaBertConfig()
        >>> model = RnaBertForCrisprOffTarget(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: RnaBertConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnabert = RnaBertModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | CrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.rnabert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.rnabert(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_attentions=sgrna_outputs.attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_attentions=target_outputs.attentions,
        )


AutoModelForCrisprOffTarget.register(RnaBertConfig, RnaBertForCrisprOffTarget)


class RnaFmForCrisprOffTarget(RnaFmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaFmConfig, RnaFmForCrisprOffTarget, RnaTokenizer
        >>> config = RnaFmConfig()
        >>> model = RnaFmForCrisprOffTarget(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: RnaFmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnafm = RnaFmModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | CrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.rnafm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.rnafm(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_attentions=sgrna_outputs.attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_attentions=target_outputs.attentions,
        )


AutoModelForCrisprOffTarget.register(RnaFmConfig, RnaFmForCrisprOffTarget)


class RnaMsmForCrisprOffTarget(RnaMsmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaMsmConfig, RnaMsmForCrisprOffTarget, RnaTokenizer
        >>> config = RnaMsmConfig()
        >>> model = RnaMsmForCrisprOffTarget(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: RnaMsmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.rnamsm = RnaMsmModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | RnaMsmForCrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.rnamsm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.rnamsm(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaMsmForCrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_col_attentions=sgrna_outputs.col_attentions,
            sgrna_row_attentions=sgrna_outputs.row_attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_col_attentions=target_outputs.col_attentions,
            target_row_attentions=target_outputs.row_attentions,
        )


AutoModelForCrisprOffTarget.register(RnaMsmConfig, RnaMsmForCrisprOffTarget)


class SpliceBertForCrisprOffTarget(SpliceBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import SpliceBertConfig, SpliceBertForCrisprOffTarget, RnaTokenizer
        >>> config = SpliceBertConfig()
        >>> model = SpliceBertForCrisprOffTarget(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: SpliceBertConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.splicebert = SpliceBertModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | CrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.splicebert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.splicebert(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_attentions=sgrna_outputs.attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_attentions=target_outputs.attentions,
        )


AutoModelForCrisprOffTarget.register(SpliceBertConfig, SpliceBertForCrisprOffTarget)


class UtrBertForCrisprOffTarget(UtrBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import UtrBertConfig, UtrBertForCrisprOffTarget, RnaTokenizer
        >>> tokenizer = RnaTokenizer(nmers=6, strameline=True)
        >>> config = UtrBertConfig(vocab_size=tokenizer.vocab_size)
        >>> model = UtrBertForCrisprOffTarget(config)
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: UtrBertConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.utrbert = UtrBertModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | CrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.utrbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.utrbert(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_attentions=sgrna_outputs.attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_attentions=target_outputs.attentions,
        )


AutoModelForCrisprOffTarget.register(UtrBertConfig, UtrBertForCrisprOffTarget)


class UtrLmForCrisprOffTarget(UtrLmPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import UtrLmConfig, UtrLmForCrisprOffTarget, RnaTokenizer
        >>> config = UtrLmConfig()
        >>> model = UtrLmForCrisprOffTarget(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> input_target = tokenizer("AUCGN", return_tensors="pt")
        >>> input["target_input_ids"] = input_target["input_ids"]
        >>> input["target_attention_mask"] = input_target["attention_mask"]
        >>> output = model(**input)
    """

    def __init__(self, config: UtrLmConfig):
        super().__init__(config)
        self.num_labels = config.head.num_labels
        self.utrlm = UtrLmModel(config, add_pooling_layer=True)
        config.head.hidden_size = config.hidden_size * 2
        self.classifier = ClassificationHead(config)
        self.head_config = self.classifier.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | CrisprOffTargetOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sgrna_outputs = self.utrlm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sgnra_embeddings = sgrna_outputs[1]
        target_outputs = self.utrlm(
            target_input_ids,
            attention_mask=target_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        target_embeddings = target_outputs[1]
        embeddings = torch.cat([sgnra_embeddings, target_embeddings], dim=-1)
        output = self.classifier(embeddings, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + sgrna_outputs[2:] + target_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrisprOffTargetOutput(
            loss=loss,
            logits=logits,
            sgrna_hidden_states=sgrna_outputs.hidden_states,
            sgrna_attentions=sgrna_outputs.attentions,
            target_hidden_states=target_outputs.hidden_states,
            target_attentions=target_outputs.attentions,
        )


AutoModelForCrisprOffTarget.register(UtrLmConfig, UtrLmForCrisprOffTarget)


@dataclass
class CrisprOffTargetOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    sgrna_hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    sgrna_attentions: Tuple[torch.FloatTensor, ...] | None = None
    target_hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    target_attentions: Tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RnaMsmForCrisprOffTargetOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    sgrna_hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    sgrna_col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    sgrna_row_attentions: Tuple[torch.FloatTensor, ...] | None = None
    target_hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    target_col_attentions: Tuple[torch.FloatTensor, ...] | None = None
    target_row_attentions: Tuple[torch.FloatTensor, ...] | None = None
