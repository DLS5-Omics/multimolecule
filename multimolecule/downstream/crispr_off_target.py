from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput

from multimolecule.models.modeling_utils import ClassificationHead
from multimolecule.models.rnabert import RnaBertConfig, RnaBertModel, RnaBertPreTrainedModel
from multimolecule.models.rnafm import RnaFmConfig, RnaFmModel, RnaFmPreTrainedModel
from multimolecule.models.rnamsm import RnaMsmConfig, RnaMsmModel, RnaMsmPreTrainedModel
from multimolecule.models.splicebert import SpliceBertConfig, SpliceBertModel, SpliceBertPreTrainedModel
from multimolecule.models.utrbert import UtrBertConfig, UtrBertModel, UtrBertPreTrainedModel
from multimolecule.models.utrlm import UtrLmConfig, UtrLmModel, UtrLmPreTrainedModel


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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
        self.sequence_head = ClassificationHead(config)
        self.head_config = self.sequence_head.config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        target_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Tensor] = None,
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
        logits = self.sequence_head(embeddings)

        loss = None
        if labels is not None:
            if self.head_config.problem_type is None:
                if self.num_labels == 1:
                    self.head_config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.head_config.problem_type = "single_label_classification"
                else:
                    self.head_config.problem_type = "multi_label_classification"
            if self.head_config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.head_config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.head_config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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


@dataclass
class CrisprOffTargetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sgrna_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    sgrna_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class RnaMsmForCrisprOffTargetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sgrna_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    sgrna_col_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sgrna_row_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_col_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_row_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
