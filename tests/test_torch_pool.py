import pytest

torch = pytest.importorskip("torch")

from biomed_rag.index.embeddings import mean_pool


def test_mean_pool_ignores_padding_and_normalizes() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(pooled[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(pooled.norm(dim=1), torch.tensor([1.0]), atol=1e-5)
