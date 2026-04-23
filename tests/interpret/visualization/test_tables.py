import torch

from multimolecule import JacobianOutput, ScalarTarget, format_topk_substitutions


def test_format_topk_substitutions_returns_dataframe():
    output = JacobianOutput(
        scores=torch.tensor([[[0.1, 0.3, 0.2], [0.4, 0.2, 0.1]]]),
        target=ScalarTarget(),
        positions=torch.tensor([2, 5]),
        top_k_indices=torch.tensor([[[1, 2], [0, 1]]]),
        top_k_scores=torch.tensor([[[0.3, 0.2], [0.4, 0.2]]]),
    )

    df = format_topk_substitutions(output, vocabulary=["A", "C", "G"])

    assert list(df.columns) == ["position", "rank", "token_index", "token", "score"]
    assert df.shape == (4, 5)
    assert df.iloc[0]["token"] == "C"
    assert df.iloc[2]["position"] == 5
