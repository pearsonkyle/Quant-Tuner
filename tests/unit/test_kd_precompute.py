"""Tests for the offline KD precompute — the architecture-agnostic bits.

These run on CPU with tiny/synthetic objects (no model downloads). What they pin:

  * ``resolve_vocab_size`` walks nested configs, so multimodal wrappers (Bee, Llava) that
    hide the LM config under ``text_config`` resolve correctly instead of raising.
  * ``_sanitize_config_dict`` coerces the float ``max_position_embeddings`` some published
    configs ship (SWE-Lego-Qwen3-8B has 163840.0), which strict validation otherwise rejects.
  * ``tokenizer_compatibility`` accepts a PADDED teacher vocab (identical tokenizers, larger
    config vocab_size — the real SWE-Lego/Bonsai case) and REFUSES genuinely divergent
    tokenizers, because per-token KD across different tokenizers is silently wrong.
  * ``kd_loss_from_topk`` is 0 when the student matches the teacher and positive otherwise.
"""

import pytest
import torch

from quant_tuner.qat.kd_precompute import (
    _sanitize_config_dict,
    kd_loss_from_topk,
    resolve_vocab_size,
    tokenizer_compatibility,
)


class _Cfg:
    """Minimal stand-in for a transformers config object."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeTok:
    def __init__(self, vocab: dict[str, int]):
        self._v = vocab

    def get_vocab(self):
        return self._v


def _vocab(n: int, prefix: str = "t") -> dict[str, int]:
    return {f"{prefix}{i}": i for i in range(n)}


# ------------------------------------------------------------------ vocab resolution ----
def test_resolve_vocab_size_flat():
    assert resolve_vocab_size(_Cfg(vocab_size=151669)) == 151669


def test_resolve_vocab_size_nested_text_config():
    # the Bee-8B-RL / Llava shape: composite arch, LM config nested
    cfg = _Cfg(vocab_size=None, text_config=_Cfg(vocab_size=151936))
    assert resolve_vocab_size(cfg) == 151936


def test_resolve_vocab_size_deeply_nested_and_failure():
    cfg = _Cfg(vocab_size=None, llm_config=_Cfg(vocab_size=None,
                                                text_config=_Cfg(vocab_size=32000)))
    assert resolve_vocab_size(cfg) == 32000
    with pytest.raises(ValueError):
        resolve_vocab_size(_Cfg(hidden_size=8))


# ------------------------------------------------------------------- config sanitizing --
def test_sanitize_coerces_float_ints_recursively():
    raw = {"max_position_embeddings": 163840.0, "hidden_size": 4096,
           "text_config": {"vocab_size": 151936.0, "num_hidden_layers": 36.0}}
    out = _sanitize_config_dict(raw)
    assert out["max_position_embeddings"] == 163840 and isinstance(
        out["max_position_embeddings"], int)
    assert out["text_config"]["vocab_size"] == 151936
    assert isinstance(out["text_config"]["num_hidden_layers"], int)
    # non-integral floats are left alone (e.g. rope theta / dropout)
    assert _sanitize_config_dict({"max_position_embeddings": 1.5})[
        "max_position_embeddings"] == 1.5


# --------------------------------------------------------------- tokenizer compatibility -
def test_padded_teacher_vocab_is_accepted():
    """The real case: identical tokenizers, teacher config vocab padded larger."""
    student = _FakeTok(_vocab(151669))
    teacher = _FakeTok(_vocab(151669))  # same tokenizer; config says 151936 (padding rows)
    n_shared, report = tokenizer_compatibility(student, teacher)
    assert n_shared == 151669
    assert "agree" in report


def test_extra_teacher_tokens_ok_when_prefix_matches():
    student = _FakeTok(_vocab(100))
    teacher = _FakeTok(_vocab(120))  # superset, shared prefix identical
    n_shared, _ = tokenizer_compatibility(student, teacher)
    assert n_shared == 100


def test_divergent_tokenizers_refused():
    student = _FakeTok(_vocab(100, prefix="a"))
    teacher = _FakeTok(_vocab(100, prefix="b"))  # same ids, different strings
    with pytest.raises(ValueError, match="disagree"):
        tokenizer_compatibility(student, teacher)


# ------------------------------------------------------------------------- the KD loss --
def test_kd_loss_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    V, P, K = 64, 5, 8
    logits = torch.randn(P, V)
    logp = torch.log_softmax(logits, dim=-1)
    vals, idx = torch.topk(logp, K, dim=-1)
    loss = kd_loss_from_topk(logits, idx, vals)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_kd_loss_positive_and_grad_flows_to_student():
    torch.manual_seed(1)
    V, P, K = 64, 5, 8
    t_logits = torch.randn(P, V)
    t_logp = torch.log_softmax(t_logits, dim=-1)
    vals, idx = torch.topk(t_logp, K, dim=-1)
    s_logits = torch.randn(P, V, requires_grad=True)
    loss = kd_loss_from_topk(s_logits, idx, vals)
    assert float(loss) > 0
    loss.backward()
    assert s_logits.grad is not None and torch.isfinite(s_logits.grad).all()


def test_kd_loss_handles_fp16_storage_roundtrip():
    """Stored logp is float16 on disk; the loss must upcast and stay finite."""
    torch.manual_seed(2)
    V, P, K = 32, 4, 6
    t_logp = torch.log_softmax(torch.randn(P, V), dim=-1)
    vals, idx = torch.topk(t_logp, K, dim=-1)
    loss = kd_loss_from_topk(torch.randn(P, V), idx.to(torch.int32), vals.to(torch.float16))
    assert torch.isfinite(loss) and float(loss) > 0
