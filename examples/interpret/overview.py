import torch
from torch import nn

from multimolecule import (
    CaLmConfig,
    CaLmForSequencePrediction,
    CaLmModel,
    attribute,
    capture_activations,
    categorical_jacobian,
    plot_token_scores,
    run_sae,
)
from multimolecule.interpret.attention import prepare_attention_output


input_ids = torch.tensor([[1, 6, 7, 2]])
tokens = ["<cls>", "A", "U", "<eos>"]


sequence_model = CaLmForSequencePrediction(CaLmConfig())

# Attribution
attribution = attribute(
    sequence_model,
    input_ids,
    method="layer_integrated_gradients",
    baseline="zero",
    n_steps=8,
)

# Activation capture
activations = capture_activations(sequence_model, input_ids, layers="embeddings")

# Attention post-processing
backbone = CaLmModel(CaLmConfig())
backbone_outputs = backbone(input_ids=input_ids, output_attentions=True, return_dict=True)
attention = prepare_attention_output(backbone_outputs.attentions, tokens=tokens, aggregate="mean_heads")

# Categorical Jacobian
jacobian = categorical_jacobian(sequence_model, input_ids, top_k=3)


class ToySae(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, feature_size)

    def encode(self, activations):
        return self.proj(activations)


sae = ToySae(hidden_size=sequence_model.config.hidden_size, feature_size=8)
sae_output = run_sae(sae, sequence_model, input_ids, layer="embeddings")

# Visualization
plot_token_scores(attribution, tokens=tokens)
