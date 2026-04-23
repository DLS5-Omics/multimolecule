import matplotlib

matplotlib.use("Agg")

import torch

from multimolecule import (
    AttentionOutput,
    AttributionOutput,
    SaeOutput,
    ScalarTarget,
    plot_attention_map,
    plot_sae_features,
    plot_token_scores,
)


def test_plot_token_scores_returns_axes():
    output = AttributionOutput(
        attributions=torch.randn(1, 4, 3),
        token_attributions=torch.tensor([[0.1, -0.2, 0.3, 0.4]]),
        method="saliency",
        target=ScalarTarget(),
        baseline="pad",
    )

    ax = plot_token_scores(output, tokens=["A", "C", "G", "U"])

    assert ax.get_title() == "Token Scores"
    assert len(ax.images) == 1


def test_plot_attention_map_supports_raw_and_rollout_outputs():
    raw = AttentionOutput(
        attentions=torch.rand(2, 1, 3, 4, 4),
        tokens=["A", "C", "G", "U"],
        layers=[0, 1],
        heads=[0, 1, 2],
        aggregation=None,
    )
    rollout = AttentionOutput(
        attentions=torch.rand(1, 4, 4),
        layers=[0, 1],
        heads=[0, 1, 2],
        aggregation="rollout",
    )

    raw_ax = plot_attention_map(raw, layer=0, head=1)
    rollout_ax = plot_attention_map(rollout, tokens=["A", "C", "G", "U"])

    assert raw_ax.get_title() == "Attention Map"
    assert rollout_ax.get_title() == "Attention Map"
    assert len(raw_ax.images) == 1
    assert len(rollout_ax.images) == 1


def test_plot_sae_features_supports_top_k():
    output = SaeOutput(
        features=torch.tensor([[[1.0, 0.1, 0.3], [0.8, 0.2, 0.4], [0.7, 0.3, 0.5]]]),
        feature_ids=torch.tensor([10, 20, 30]),
    )

    ax = plot_sae_features(output, top_k=2)

    assert ax.get_title() == "SAE Features"
    assert len(ax.images) == 1
    assert len(ax.get_yticklabels()) == 2
