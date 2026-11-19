# Model interpretation

This quickstart compares two ways to attribute an RNA-FM prediction to its input tokens. It runs on CPU with a small, randomly initialized model and two synthetic sequences.

## Run the example

From the repository root, install the optional interpretation dependencies and run the script.

```bash
python -m pip install -e ".[interpret]"
python examples/interpret/quickstart.py
```

`output="logits"` selects the model output tensor. `target=1` selects its second logit for each sample. The script saves `interpret-quickstart.png` in the current directory, showing the first sequence with both methods on a shared color scale centered at zero.

```python
--8<-- "examples/interpret/quickstart.py:23:"
```

## Read the scores

- Use `layer_integrated_gradients` to attribute a score along a differentiable path between the baseline and input embeddings. It returns `[B, L, D]`, which the example sums over the embedding dimension to obtain `[B, L]` signed token scores. The convergence delta measures how closely their sum matches the logit difference between the input and its baseline.
- Use `feature_ablation` to subtract the logit after replacing one token from the logit of the original sequence. It works directly on integer token IDs and returns `[B, L]` in this example. Interactions between tokens can contribute to several ablation scores, so the two methods can report different values.
- Use `model.eval()` for evaluation behavior such as disabling dropout. Run Layer Integrated Gradients outside `torch.inference_mode()` so gradients are available.
- The baseline replaces sequence tokens with `<unk>` and preserves special tokens, padding and the attention mask. Replacing a sequence token with `<pad>` can also change RNA-FM's automatically inferred position IDs. The script uses `special_tokens_mask` to choose reference and plot positions, and passes only `input_ids` and `attention_mask` to the model.
