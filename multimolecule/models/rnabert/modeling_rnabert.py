import math

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .configuration_rnabert import RnaBertConfig


class RnaBertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RnaBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RnaBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RnaBertAttention(config)
        self.intermediate = RnaBertIntermediate(config)
        self.output = RnaBertOutput(config)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output, outputs = self_attention_outputs[0], self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class RnaBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.selfattn = RnaBertSelfAttention(config)
        self.output = RnaBertSelfOutput(config)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        self_outputs = self.selfattn(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RnaBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads': 12

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class RnaBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RnaBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RnaBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RnaBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([RnaBertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.layer = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size)
        #                             for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for layer in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class RnaBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)

        pooled_output = self.activation(pooled_output)

        return pooled_output


class RnaBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RnaBertConfig
    base_model_prefix = "rnabert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RnaBertLayer", "RnaBertFoldTriangularSelfAttentionBlock", "RnaBertEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RnaBertModel(RnaBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RnaBertEmbeddings(config)
        self.encoder = RnaBertEncoder(config)
        self.pooler = RnaBertPooler(config)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            # attention_mask=attention_mask,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RnaBertLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.predictions = MaskedWordPredictions(config)
        config.vocab_size = config.ss_size
        self.predictions_ss = MaskedWordPredictions(config)

        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        prediction_scores_ss = self.predictions_ss(sequence_output)

        seq_relationship_score = self.seq_relationship(pooled_output)

        return prediction_scores, prediction_scores_ss, seq_relationship_score


class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transform = RnaBertPredictionHeadTransform(config)

        self.decoder = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class RnaBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.transform_act_fn = gelu

        self.LayerNorm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RnaBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = RnaBertModel(config)
        self.lm_head = RnaBertLMHead(config)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        prediction_scores, prediction_scores_ss, seq_relationship_score = self.lm_head(
            outputs.last_hidden_state, outputs.pooler_output
        )
        return prediction_scores, prediction_scores_ss, outputs
