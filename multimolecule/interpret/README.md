# interpret

`interpret` provides model interpretation utilities for sequence models in MultiMolecule.

It currently includes:

- `attribute(...)` for Captum-based attribution
- `capture_activations(...)` for activation capture
- attention tensor post-processing helpers in `multimolecule.interpret.attention`
- `categorical_jacobian(...)` for categorical Jacobian analysis
- `run_sae(...)` for running external sparse autoencoders on captured activations
- plotting and tabulation helpers in `multimolecule.interpret.visualization`

## Optional Dependencies

Some interpretation features depend on optional packages:

- `captum` is required for attribution methods
- `matplotlib` is required for plotting helpers

These dependencies use deferred failure: importing `multimolecule` does not require them, but calling the affected features does.

## Usage

### Attribution

```python
import torch
from multimolecule import CaLmConfig, CaLmForSequencePrediction, attribute

model = CaLmForSequencePrediction(CaLmConfig())
input_ids = torch.tensor([[1, 6, 7, 2]])

output = attribute(
    model,
    input_ids,
    method="layer_integrated_gradients",
    baseline="zero",
    n_steps=8,
)

print(output.token_attributions.shape)
```

### Activation Capture

```python
import torch
from multimolecule import CaLmConfig, CaLmForSequencePrediction, capture_activations

model = CaLmForSequencePrediction(CaLmConfig())
input_ids = torch.tensor([[1, 6, 7, 2]])

output = capture_activations(model, input_ids, layers="embeddings")
print(output.resolved_layers)
```

### Attention Analysis

```python
import torch
from multimolecule import CaLmConfig, CaLmModel
from multimolecule.interpret.attention import prepare_attention_output

model = CaLmModel(CaLmConfig())
input_ids = torch.tensor([[1, 6, 7, 2]])

outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)
attention = prepare_attention_output(outputs.attentions, aggregate="mean_heads")
print(attention.attentions.shape)
```

### Categorical Jacobian

```python
import torch
from multimolecule import CaLmConfig, CaLmForSequencePrediction, categorical_jacobian

model = CaLmForSequencePrediction(CaLmConfig())
input_ids = torch.tensor([[1, 6, 7, 2]])

output = categorical_jacobian(model, input_ids, top_k=3)
print(output.top_k_indices.shape)
```

### SAE Adapter

```python
import torch
from torch import nn
from multimolecule import CaLmConfig, CaLmForSequencePrediction, run_sae


class ToySae(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, feature_size)

    def encode(self, activations):
        return self.proj(activations)


model = CaLmForSequencePrediction(CaLmConfig())
sae = ToySae(hidden_size=model.config.hidden_size, feature_size=8)
input_ids = torch.tensor([[1, 6, 7, 2]])

output = run_sae(sae, model, input_ids, layer="embeddings")
print(output.features.shape)
```

### Visualization

```python
from multimolecule import plot_token_scores

ax = plot_token_scores(output, tokens=["<cls>", "A", "U", "<eos>"])
```
