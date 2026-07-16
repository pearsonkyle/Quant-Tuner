"""Tests for the QAT trainer (scripts/exp058_qat_train_v2.py) + master_opt.

Everything runs on CPU with a tiny random-init Qwen3 (hidden 128 — the smallest
width the g128 ternarizer accepts). The load-bearing checks:

  * masked-CE parity: the gather-then-lm_head loss must equal transformers'
    full-logits ForCausalLMLoss bit-for-bit-ish (same target set, same mean),
    or the grad-accum semantics silently change.
  * wrap_model: frozen layers with exactly-ternary weights are left UNWRAPPED
    (fast path) without changing the trainable param set or the step-0 logits;
    off-grid frozen weights fall back to wrapping.
  * resume: fingerprint-guarded round-trip through a real (tiny) training run.
  * MasterOptimizer: identical to the plain optimizer at fp32, and demonstrably
    fixes the bf16 sub-ulp underflow (the "no codes flip" failure).
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

from quant_tuner.qat import train as trainer  # noqa: E402
from quant_tuner.qat.master_opt import MasterOptimizer  # noqa: E402
from quant_tuner.qat.ternary import TernaryLinear, ternarize_group  # noqa: E402

VOCAB = 512
HID = 128


def tiny_model(seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(hidden_size=HID, intermediate_size=2 * HID, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                      vocab_size=VOCAB, max_position_embeddings=256,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg)


def make_ternary(model) -> None:
    """Snap every wrappable linear onto the exact ternary grid (shipped-style)."""
    with torch.no_grad():
        for layer in model.model.layers:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear) and m.in_features % 128 == 0:
                    _, _, w_hat = ternarize_group(m.weight)
                    m.weight.copy_(w_hat)


def rand_batch(seed: int = 1, seq: int = 48, frac_labeled: float = 0.4):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, seq), generator=g)
    lbl = ids.clone()
    mask = torch.rand(1, seq, generator=g) > frac_labeled
    lbl[mask] = -100
    return ids, lbl


# ---------------------------------------------------------------- masked CE --

def test_masked_ce_parity_with_hf_full_logits():
    model = tiny_model().eval()
    ids, lbl = rand_batch()
    with torch.no_grad():
        ref = model(input_ids=ids, labels=lbl).loss
        ce, logits, keep_idx = trainer.masked_forward(model, ids, lbl)
    assert logits.shape == (1, int(keep_idx.numel()), VOCAB)
    assert torch.allclose(ce, ref, atol=1e-6), f"{ce} vs HF {ref}"


def test_masked_ce_parity_single_target():
    model = tiny_model().eval()
    ids, _ = rand_batch(seed=3)
    lbl = torch.full_like(ids, -100)
    lbl[0, 5] = ids[0, 5]  # exactly one shifted target (predicted from pos 4)
    with torch.no_grad():
        ref = model(input_ids=ids, labels=lbl).loss
        ce, _, keep_idx = trainer.masked_forward(model, ids, lbl)
    assert keep_idx.tolist() == [4]
    assert torch.allclose(ce, ref, atol=1e-6)


# ---------------------------------------------------------------- wrap_model --

def test_wrap_model_frozen_exact_layers_left_unwrapped():
    model = tiny_model()
    make_ternary(model)
    trainer.wrap_model(model, 0, layer_spec="0")  # train layer 0 only
    l0 = model.model.layers[0]
    l1 = model.model.layers[1]
    assert any(isinstance(m, TernaryLinear) for m in l0.modules())
    assert not any(isinstance(m, TernaryLinear) for m in l1.modules())  # fast path
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".linear.weight" in n and "layers.0." in n for n in trainable)


def test_wrap_model_offgrid_frozen_falls_back_to_wrapping():
    model = tiny_model()  # random weights: NOT on the ternary grid
    trainer.wrap_model(model, 0, layer_spec="0")
    l1 = model.model.layers[1]
    assert any(isinstance(m, TernaryLinear) for m in l1.modules())  # wrapped anyway
    # ...but still frozen
    assert not any(p.requires_grad for p in l1.parameters())


def test_wrap_model_train_norms_extends_filter():
    model = tiny_model()
    make_ternary(model)
    trainer.wrap_model(model, 0, layer_spec="0", train_norms=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("input_layernorm" in n for n in trainable)
    assert any("q_norm" in n for n in trainable)
    assert all("layers.0." in n for n in trainable)  # never layer 1 / final norm
    assert not any("model.norm" in n for n in trainable)


def test_step0_logits_identity_wrapped_vs_unwrapped_frozen():
    model = tiny_model().eval()
    make_ternary(model)
    ids, _ = rand_batch(seed=5)
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()
    trainer.wrap_model(model, 0, layer_spec="0")
    model.eval()
    with torch.no_grad():
        got = model(input_ids=ids).logits
    assert torch.equal(got, ref), "step-0 forward must be bit-identical after wrapping"


# ------------------------------------------------------------ lr / layers ----

def test_lr_schedule_pinned():
    assert trainer.lr_at(0, 100, 1.0) == 0.0
    assert trainer.lr_at(5, 100, 1.0) == 1.0  # warmup end (5% of 100)
    assert math.isclose(trainer.lr_at(100, 100, 1.0), 0.1, rel_tol=1e-6)  # cosine floor


def test_parse_layers():
    assert trainer.parse_layers("0-14,32,34,35", 36) == set(range(15)) | {32, 34, 35}
    assert trainer.parse_layers("40", 36) == set()  # out of range dropped
    assert trainer.parse_layers("", 36) == set()


# ------------------------------------------------------------- telemetry -----

def test_flip_telemetry_detects_code_flips_and_scale_drift():
    model = tiny_model()
    make_ternary(model)
    trainer.wrap_model(model, 2, layer_spec="0-1")
    snaps = trainer.snapshot_codes(model, k=2)
    assert len(snaps) == 2
    stats, _ = trainer.flip_report(model, snaps)
    assert all(s["flip_pct"] == 0.0 and s["scale_drift"] < 1e-6 for s in stats.values())
    # zero out one group and rescale another -> flips + drift register
    name = next(iter(snaps))
    w = dict(model.named_modules())[name].linear.weight
    with torch.no_grad():
        w[0, :128] = 0.0
        w[1] *= 1.5
    stats, _ = trainer.flip_report(model, snaps)
    assert stats[name]["nonzero_to_zero"] > 0
    assert stats[name]["scale_drift"] > 0.01


# -------------------------------------------------------- MasterOptimizer ----

def test_master_optimizer_matches_plain_at_fp32():
    torch.manual_seed(0)
    w0 = torch.randn(8, 8)
    pa = torch.nn.Parameter(w0.clone())
    pb = torch.nn.Parameter(w0.clone())
    plain = torch.optim.AdamW([pa], lr=1e-2, weight_decay=0.0, foreach=False)
    wrapped = MasterOptimizer([pb], lambda ms: torch.optim.AdamW(
        ms, lr=1e-2, weight_decay=0.0, foreach=False))
    for i in range(3):
        gr = torch.randn(8, 8, generator=torch.Generator().manual_seed(i))
        pa.grad = gr.clone()
        pb.grad = gr.clone()
        torch.nn.utils.clip_grad_norm_([pa], 1.0, foreach=False)
        plain.step()
        plain.zero_grad()
        wrapped.clip_and_step(1.0)
    assert torch.allclose(pa, pb, atol=1e-7)


def test_master_optimizer_fixes_bf16_underflow():
    # bf16 ulp at 1.0 is 2^-8 ~ 3.9e-3: a 1e-3 SGD update vanishes in raw bf16
    # (the "no codes flip" failure) but accumulates in the fp32 master.
    raw = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    sgd = torch.optim.SGD([raw], lr=1e-3)
    for _ in range(20):
        raw.grad = torch.ones_like(raw)
        sgd.step()
        sgd.zero_grad()
    assert torch.equal(raw.detach(), torch.ones(4, dtype=torch.bfloat16))  # never moved

    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = MasterOptimizer([p], lambda ms: torch.optim.SGD(ms, lr=1e-3))
    for _ in range(20):
        p.grad = torch.ones_like(p)
        opt.clip_and_step(max_norm=None)
    assert not torch.equal(p.detach(), torch.ones(4, dtype=torch.bfloat16))
    assert torch.allclose(opt.masters[0].detach(), torch.full((4,), 0.98), atol=1e-4)


def test_master_optimizer_state_roundtrip():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = MasterOptimizer([p], lambda ms: torch.optim.SGD(ms, lr=1e-3))
    p.grad = torch.ones_like(p)
    opt.clip_and_step(max_norm=None)
    sd = opt.state_dict()
    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt2 = MasterOptimizer([p2], lambda ms: torch.optim.SGD(ms, lr=1e-3))
    opt2.load_state_dict(sd)
    assert torch.equal(opt2.masters[0].detach(), opt.masters[0].detach())
    assert torch.equal(p2.detach(), p.detach())


# ----------------------------------------------------------------- KD --------

def test_kd_kl_zero_for_identical_teacher_and_grads_only_to_student():
    model = tiny_model().eval()
    ids, lbl = rand_batch(seed=7)
    ce, s_logits, keep_idx = trainer.masked_forward(model, ids, lbl)
    kl = trainer.kd_kl(model, ids, keep_idx, s_logits, temp=1.0)
    assert float(kl.detach()) < 1e-5  # teacher == student -> ~0
    loss = 0.5 * ce + 0.5 * kl
    loss.backward()  # must not try to backprop through the (no_grad) teacher pass
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------- end-to-end + resume --

def _write_tiny_corpus(path: Path, n_win: int = 8, seq: int = 48, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n_win, seq), generator=g)
    lbl = ids.clone()
    lbl[torch.rand(n_win, seq, generator=g) > 0.5] = -100
    lbl[:, -8:-4] = ids[:, -8:-4]  # guarantee >=4 shifted targets per window
    fp = trainer.corpus_fingerprint(ids, lbl)
    torch.save({"ids": ids, "labels": lbl, "window": seq, "assistant_frac": 0.5,
                "fingerprint": fp}, path)
    return fp


def _run_main(monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["exp058_qat_train_v2.py", *argv])
    return trainer.main()


@pytest.fixture()
def tiny_env(tmp_path, monkeypatch):
    mdir = tmp_path / "model"
    m = tiny_model(seed=42)
    make_ternary(m)
    m.save_pretrained(mdir)
    monkeypatch.setattr(trainer, "MODEL", mdir)
    corpus = tmp_path / "corpus.pt"
    _write_tiny_corpus(corpus)
    return tmp_path, corpus


def test_train_resume_roundtrip(tiny_env, monkeypatch):
    tmp_path, corpus = tiny_env
    out = tmp_path / "trained"
    base = ["--corpus", str(corpus), "--layers", "0-1", "--grad-accum", "2",
            "--lr", "1e-3", "--optim", "adafactor", "--ckpt-every", "2",
            "--flip-sample", "2", "--out", str(out)]
    assert _run_main(monkeypatch, [*base, "--epochs", "1"]) == 0  # 8 win / 2 = 4 steps
    ck = torch.load(out / "trained_latents.pt", weights_only=False)
    assert ck["step"] == 4 and ck["mi"] == 8
    assert ck["optim"] is not None and ck["corpus_fingerprint"]
    assert all(".linear.weight" in k for k in ck["latents"])
    assert ck["loss_first"] is not None and ck["loss_last"] is not None
    w0 = next(iter(ck["latents"].values())).clone()

    # resume for a second epoch: continues from step 4 -> 8, latents keep moving
    assert _run_main(monkeypatch,
                     [*base, "--epochs", "2", "--resume", str(out / "trained_latents.pt")]) == 0
    ck2 = torch.load(out / "trained_latents.pt", weights_only=False)
    assert ck2["step"] == 8 and ck2["mi"] == 16
    assert not torch.equal(next(iter(ck2["latents"].values())), w0)


def test_resume_rejects_rebuilt_corpus(tiny_env, monkeypatch):
    tmp_path, corpus = tiny_env
    out = tmp_path / "trained"
    base = ["--corpus", str(corpus), "--layers", "0-1", "--grad-accum", "2",
            "--lr", "1e-3", "--ckpt-every", "0", "--flip-sample", "0", "--out", str(out)]
    assert _run_main(monkeypatch, [*base, "--epochs", "0.5"]) == 0
    _write_tiny_corpus(corpus, seed=99)  # rebuild -> new fingerprint
    with pytest.raises(SystemExit, match="corpus mismatch"):
        _run_main(monkeypatch, [*base, "--epochs", "1",
                                "--resume", str(out / "trained_latents.pt")])


def test_nonfinite_window_skipped_without_breaking_accum(tiny_env, monkeypatch):
    tmp_path, corpus = tiny_env
    out = tmp_path / "trained_nan"
    orig = trainer.masked_forward
    calls = {"n": 0}

    def flaky(model, ids, lbl):
        ce, logits, keep = orig(model, ids, lbl)
        calls["n"] += 1
        if calls["n"] == 1:
            return ce * float("nan"), logits, keep
        return ce, logits, keep

    monkeypatch.setattr(trainer, "masked_forward", flaky)
    assert _run_main(monkeypatch, ["--corpus", str(corpus), "--layers", "0-1",
                                   "--grad-accum", "2", "--epochs", "1",
                                   "--ckpt-every", "0", "--flip-sample", "0",
                                   "--out", str(out)]) == 0
    ck = torch.load(out / "trained_latents.pt", weights_only=False)
    assert ck["step"] == 4  # NaN window skipped; every step still saw 2 clean windows
    assert calls["n"] >= 9  # one extra window consumed to replace the bad one


def test_json_serializable_args_in_ckpt(tiny_env, monkeypatch):
    tmp_path, corpus = tiny_env
    out = tmp_path / "trained_args"
    assert _run_main(monkeypatch, ["--corpus", str(corpus), "--layers", "0",
                                   "--grad-accum", "2", "--epochs", "0.5",
                                   "--ckpt-every", "0", "--flip-sample", "0",
                                   "--out", str(out)]) == 0
    ck = torch.load(out / "trained_latents.pt", weights_only=False)
    json.dumps(ck["args"])  # stringified Paths etc. must survive
