"""Unit tests for calibrate._ingest — strided corpus chunk sampling."""

from pathlib import Path

import pytest
import torch

from quant_tuner.calibrate._ingest import load_chunks, sample_chunks, sample_rows


def _ids(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long)


def test_budget_covers_corpus_returns_all_chunks_in_order():
    chunks = sample_chunks(_ids(100), ctx=10, budget_tokens=100)
    assert len(chunks) == 10
    assert torch.equal(torch.cat(chunks), _ids(100))


def test_stride_includes_first_and_last_chunk():
    # 100 chunks of 10 tokens, budget for 3 chunks.
    chunks = sample_chunks(_ids(1000), ctx=10, budget_tokens=30)
    assert len(chunks) == 3
    assert chunks[0][0].item() == 0        # first chunk
    assert chunks[-1][-1].item() == 999    # last chunk
    # Middle pick is ~evenly spaced, not adjacent to either end.
    assert 300 < chunks[1][0].item() < 700


def test_budget_respected_not_head_truncated():
    # Head-truncation would return chunks 0..3; the stride must reach the tail.
    chunks = sample_chunks(_ids(10_000), ctx=100, budget_tokens=400)
    assert len(chunks) == 4
    starts = [c[0].item() for c in chunks]
    assert starts[0] == 0
    assert starts[-1] == 9_900
    assert starts == sorted(starts)


def test_short_corpus_passthrough():
    chunks = sample_chunks(_ids(7), ctx=100, budget_tokens=1_000_000)
    assert len(chunks) == 1
    assert chunks[0].numel() == 7


def test_trailing_sliver_dropped():
    # 101 tokens @ctx 10 -> the 1-token tail chunk is unusable for a forward.
    chunks = sample_chunks(_ids(101), ctx=10, budget_tokens=10_000)
    assert all(c.numel() >= 2 for c in chunks)
    assert len(chunks) == 10


def test_deterministic():
    a = sample_chunks(_ids(5000), ctx=64, budget_tokens=256)
    b = sample_chunks(_ids(5000), ctx=64, budget_tokens=256)
    assert len(a) == len(b)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        sample_chunks(_ids(10), ctx=1, budget_tokens=10)
    with pytest.raises(ValueError):
        sample_chunks(_ids(10), ctx=4, budget_tokens=0)


def test_empty_corpus_returns_no_chunks():
    assert sample_chunks(_ids(0), ctx=8, budget_tokens=8) == []
    assert sample_chunks(_ids(1), ctx=8, budget_tokens=8) == []


def test_sample_rows_even_spread_and_passthrough():
    x = torch.arange(100).unsqueeze(1).float()
    picked = sample_rows(x, 5)
    assert picked.shape == (5, 1)
    assert picked[0, 0].item() == 0.0
    assert picked[-1, 0].item() == 99.0
    # n >= len returns the tensor unchanged.
    assert sample_rows(x, 200) is x


class _FakeTok:
    """1 token per whitespace-separated word, ids = word index."""

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        n = len(text.split())

        class R:
            input_ids = torch.arange(n, dtype=torch.long).unsqueeze(0)

        return R()


def test_load_chunks_reads_whole_file_and_strides(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(f"w{i}" for i in range(1000)))
    chunks = load_chunks(_FakeTok(), corpus, ctx=10, budget_tokens=30)
    assert len(chunks) == 3
    # The old head-truncation could never produce a token id past budget_tokens.
    assert chunks[-1].max().item() == 999
