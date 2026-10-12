import pytest
import torch

from multimolecule.tokenisers.utils import convert_word_embeddings


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_convert_word_embeddings_preserves_source_values(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    weight = torch.tensor([[1.000000000000123, 2.0], [3.0, 4.0]], dtype=dtype, device=device)
    bias = torch.tensor([1.000000000000123, 3.0], dtype=dtype, device=device)
    converted = convert_word_embeddings(
        weight, bias, old_vocab=["A", "B"], new_vocab=["B", "A", "X"], vocab_mapping={"X": "AB"}
    )
    for source, target in zip((weight, bias), converted):
        assert target.dtype == source.dtype
        assert target.device == source.device
        torch.testing.assert_close(target[:2], source.flip(0), rtol=0, atol=0)
        torch.testing.assert_close(target[2], source.mean(0))
