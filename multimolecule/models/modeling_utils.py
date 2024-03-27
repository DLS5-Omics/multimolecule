from typing import Optional, Tuple, Union

import torch
from chanfig import ConfigRegistry
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput


class MaskedLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        if "proj_head_mode" not in dir(config) or config.proj_head_mode is None:
            config.proj_head_mode = "none"
        self.transform = PredictionHeadTransform.build(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        x = self.transform(sequence_output)
        prediction_scores = self.decoder(x)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SequenceClassificationHead(nn.Module):
    """Head for sequence-level classification tasks."""

    num_labels: int

    def __init__(self, config):
        super().__init__()
        if "proj_head_mode" not in dir(config) or config.proj_head_mode is None:
            config.proj_head_mode = "none"
        self.num_labels = config.num_labels
        self.transform = PredictionHeadTransform.build(config)
        classifier_dropout = (
            config.classifier_dropout
            if "classifier_dropout" in dir(config) and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def forward(
        self, outputs, labels: Optional[torch.Tensor] = None, return_dict: Optional[bool] = None
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        x = self.dropout(sequence_output)
        x = self.transform(x)
        logits = self.decoder(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TokenClassificationHead(nn.Module):
    """Head for token-level classification tasks."""

    num_labels: int

    def __init__(self, config):
        if "proj_head_mode" not in dir(config) or config.proj_head_mode is None:
            config.proj_head_mode = "none"
        super().__init__()
        self.num_labels = config.num_labels
        self.transform = PredictionHeadTransform.build(config)
        classifier_dropout = (
            config.classifier_dropout
            if "classifier_dropout" in dir(config) and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.decoder = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def forward(
        self, outputs, labels: Optional[torch.Tensor] = None, return_dict: Optional[bool] = None
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        token_output = outputs.pooled_output if return_dict else outputs[1]
        x = self.dropout(token_output)
        x = self.transform(x)
        logits = self.decoder(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


PredictionHeadTransform = ConfigRegistry(key="proj_head_mode")


@PredictionHeadTransform.register("nonlinear")
class NonLinearTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register("linear")
class LinearTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


@PredictionHeadTransform.register("none")
class IdentityTransform(nn.Identity):
    def __init__(self, config):
        super().__init__()
