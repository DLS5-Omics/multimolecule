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


import matplotlib.pyplot as plt
import torch

from multimolecule import RnaFmConfig, RnaFmForSequencePrediction, RnaTokenizer
from multimolecule.interpret import attribution
from multimolecule.visualize.interpret import plot_token_scores


def main() -> None:
    torch.manual_seed(12)
    tokenizer = RnaTokenizer()
    model = RnaFmForSequencePrediction(
        RnaFmConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            num_labels=2,
        )
    ).eval()

    encoded = tokenizer(["ACGU", "AGCUAC"], padding=True, return_tensors="pt", return_special_tokens_mask=True)
    batch = {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask}
    sequence_tokens = encoded.attention_mask.bool() & ~encoded.special_tokens_mask.bool()
    baseline = encoded.input_ids.masked_fill(sequence_tokens, tokenizer.unk_token_id)

    target = 1
    layer_ig = attribution.layer_integrated_gradients(
        model,
        batch,
        input_keys="input_ids",
        layer=model.get_input_embeddings(),
        baselines=baseline,
        output="logits",
        target=target,
        n_steps=32,
        internal_batch_size=8,
        return_convergence_delta=True,
    )
    ablation = attribution.feature_ablation(
        model,
        batch,
        input_keys="input_ids",
        baselines=baseline,
        output="logits",
        target=target,
        perturbations_per_eval=4,
    )

    layer_scores = layer_ig.attributions.sum(dim=-1)
    ablation_scores = ablation.attributions
    with torch.no_grad():
        logits = model(**batch).logits
        reference_logits = model(input_ids=baseline, attention_mask=batch["attention_mask"]).logits
    print("Layer IG attribution shape", tuple(layer_ig.attributions.shape))
    print("Feature Ablation attribution shape", tuple(ablation_scores.shape))
    print("Target logit difference", (logits[:, target] - reference_logits[:, target]).tolist())
    print("Layer IG token score sum", layer_scores.sum(dim=-1).tolist())
    print("Layer IG convergence delta", layer_ig.convergence_delta.tolist())

    sample = 0
    positions = sequence_tokens[sample]
    labels = tokenizer.convert_ids_to_tokens(encoded.input_ids[sample, positions].tolist())
    ig_values = layer_scores[sample, positions]
    ablation_values = ablation_scores[sample, positions]

    fig, axes = plt.subplots(2, 1, figsize=(7, 3), constrained_layout=True)
    plot_token_scores(ig_values, tokens=labels, ax=axes[0], title="Layer Integrated Gradients")
    plot_token_scores(ablation_values, tokens=labels, ax=axes[1], title="Feature Ablation")
    limit = max(ig_values.abs().max().item(), ablation_values.abs().max().item())
    if limit > 0:
        for ax in axes:
            ax.images[0].set_clim(-limit, limit)

    fig.savefig("interpret-quickstart.png", dpi=160)
    plt.close(fig)
    print("Saved interpret-quickstart.png")


if __name__ == "__main__":
    main()
