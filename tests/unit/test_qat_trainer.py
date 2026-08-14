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

    def flaky(model, ids, lbl, **kw):
        ce, logits, keep = orig(model, ids, lbl, **kw)
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


def test_chunked_masked_ce_matches_unchunked_loss_and_grads():
    """need_logits=False chunks the lm_head to cap a K-dependent multi-GB spike.

    It must be the SAME loss: a mean over all K, not a mean of per-chunk means (those
    differ whenever K is not a multiple of the chunk — which is the common case). At
    K=8064/V=151669 the unchunked logits alone are 4.6 GB, and K varies per window, so
    this is what stops an intermittent OOM at an unpredictable step.
    """
    model = tiny_model()
    ids, lbl = rand_batch(seed=11)
    k = int((lbl[0, 1:] != -100).sum())
    assert k % 5 != 0, "pick a batch where K is NOT a multiple of the chunk"

    ce_ref, logits, keep_ref = trainer.masked_forward(model, ids, lbl)
    ce_ref.backward()
    g_ref = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)

    ce_chunk, none_logits, keep_chunk = trainer.masked_forward(
        model, ids, lbl, need_logits=False, logit_chunk=5)
    ce_chunk.backward()
    g_chunk = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert none_logits is None                       # caller must not rely on them
    assert logits is not None
    assert torch.equal(keep_ref, keep_chunk)
    assert torch.allclose(ce_ref, ce_chunk, atol=1e-6), (float(ce_ref), float(ce_chunk))
    assert len(g_chunk) == len(g_ref) and g_ref
    for a, b in zip(g_ref, g_chunk, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_chunked_masked_ce_still_matches_hf_full_logits():
    """The chunked path is the one training actually uses — hold it to the HF loss too."""
    model = tiny_model().eval()
    ids, lbl = rand_batch(seed=13)
    with torch.no_grad():
        ref = model(input_ids=ids, labels=lbl).loss
        ce, logits, _ = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                               logit_chunk=4)
    assert logits is None
    assert torch.allclose(ce, ref, atol=1e-6), f"{ce} vs HF {ref}"


def test_chunking_is_skipped_when_k_fits_in_one_chunk():
    model = tiny_model().eval()
    ids, _ = rand_batch(seed=3)
    lbl = torch.full_like(ids, -100)
    lbl[0, 5] = ids[0, 5]                            # K = 1
    with torch.no_grad():
        ref = model(input_ids=ids, labels=lbl).loss
        ce, _, _ = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                          logit_chunk=1024)
    assert torch.allclose(ce, ref, atol=1e-6)


# ------------------------------------------------- prefix context / trained tail --
#
# `--trained-tail` is the lever that makes a 32768 window fit: the prefix is encoded once
# under no_grad into a KV cache, so activation memory tracks the tail, not the window.
# The correctness claim is that the tail's loss is UNCHANGED by the split — if it were
# not, every long-window result would be measuring a different objective.

@pytest.fixture
def sdpa_patched():
    """The prefix K/V ride in the patched attention function, so these need it on."""
    from quant_tuner.qat.attention import clear_prefix, disable_chunked_sdpa, enable_chunked_sdpa
    enable_chunked_sdpa()
    yield
    clear_prefix()
    disable_chunked_sdpa()


def test_prefix_context_tail_loss_equals_the_full_window_loss(sdpa_patched):
    model = tiny_model().eval()
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    n_prefix = 32
    tail_only = lbl.clone()
    tail_only[0, :n_prefix + 1] = -100  # the targets a prefix split keeps (shifted)
    with torch.no_grad():
        ref, _, ref_idx = trainer.masked_forward(model, ids, tail_only, need_logits=False)
        with trainer.prefix_window(model, ids, n_prefix):
            got, _, got_idx = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                                     n_prefix=n_prefix)
    assert torch.equal(ref_idx, got_idx)
    assert torch.allclose(got, ref, atol=1e-5), f"{got} vs full-window {ref}"


def test_prefix_context_drops_prefix_targets_from_the_loss(sdpa_patched):
    """A prefix target has no graph, so it must not be scored — and the caller must be
    able to tell, because the resulting CE is on a different target set."""
    model = tiny_model().eval()
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    with torch.no_grad():
        _, _, full_idx = trainer.masked_forward(model, ids, lbl, need_logits=False)
        with trainer.prefix_window(model, ids, 32):
            _, _, tail_idx = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                                    n_prefix=32)
    assert tail_idx.numel() < full_idx.numel()
    assert int(tail_idx.min()) >= 32


def test_prefix_covering_every_target_raises_rather_than_returning_zero(sdpa_patched):
    model = tiny_model().eval()
    ids, _ = rand_batch(seq=64)
    lbl = torch.full_like(ids, -100)
    lbl[0, 5] = ids[0, 5]
    with pytest.raises(ValueError, match="no labeled target"), torch.no_grad(), \
            trainer.prefix_window(model, ids, 32):
        trainer.masked_forward(model, ids, lbl, need_logits=False, n_prefix=32)


def test_prefix_gradients_reach_the_tail_only(sdpa_patched):
    model = tiny_model()
    make_ternary(model)
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    emb = model.model.embed_tokens.weight
    with trainer.prefix_window(model, ids, 32):
        ce, _, _ = trainer.masked_forward(model, ids, lbl, need_logits=False, n_prefix=32)
        ce.backward()
    # embeddings of prefix-only token positions get no gradient through the frozen prefix
    assert emb.grad is not None and emb.grad.abs().sum() > 0


def test_prefix_survives_gradient_checkpointing(sdpa_patched):
    """The reason the prefix does NOT use a transformers Cache: GradientCheckpointingLayer
    nulls `past_key_values` whenever `gradient_checkpointing and training`, so a cache-based
    tail attends to nothing and the loss still falls. Riding in the attention function is
    immune — this pins that."""
    model = tiny_model()
    make_ternary(model)
    model.gradient_checkpointing_enable()
    model.train()
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    with trainer.prefix_window(model, ids, 32):
        with_prefix, _, _ = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                                   n_prefix=32)
        # backward inside the block: checkpoint recompute needs the prefix still live
        with_prefix.backward()
    tail_only = lbl.clone()
    tail_only[0, :33] = -100
    model.eval()
    with torch.no_grad():
        ref, _, _ = trainer.masked_forward(model, ids, tail_only, need_logits=False)
    assert torch.allclose(with_prefix.detach(), ref, atol=1e-5), (
        f"{with_prefix.item()} vs full-context {ref.item()} — the prefix was dropped")


def test_use_prefix_without_a_capture_raises(sdpa_patched):
    from quant_tuner.qat.attention import use_prefix
    with pytest.raises(RuntimeError, match="nothing captured"), use_prefix():
        pass


# ----------------------------------------------------------- stop-token weighting --

def test_stop_weight_reweights_the_loss_and_its_denominator():
    """sum(w[target]) must be the denominator: with plain K the loss (and the effective
    LR) would scale with however often the stop token happened to land in the window."""
    model = tiny_model().eval()
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    shifted = lbl[0, 1:]
    stop_id = int(shifted[shifted != -100][0])
    w = torch.ones(VOCAB)
    w[stop_id] = 7.0
    with torch.no_grad():
        weighted, _, keep = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                                   logit_chunk=8, weights=w)
        # reference: sum(w_i * l_i) / sum(w_i) over exactly the same targets. Stated as
        # per-target losses rather than F.cross_entropy(weight=) so the denominator is
        # asserted explicitly — that is the part a chunked loop gets wrong.
        tgts = lbl[0, 1:][keep]
        logits = model.lm_head(model.model(input_ids=ids).last_hidden_state[:, keep, :])
        per = torch.nn.functional.cross_entropy(logits[0].float(), tgts, reduction="none")
        ref = (per * w[tgts]).sum() / w[tgts].sum()
    assert torch.allclose(weighted, ref, atol=1e-5), f"{weighted} vs {ref}"
    # and the weight actually lands on the intended id
    assert (w[tgts] == 7.0).any()


def test_stop_weight_of_one_is_a_no_op():
    model = tiny_model().eval()
    ids, lbl = rand_batch(seq=64, frac_labeled=0.5)
    with torch.no_grad():
        plain, _, _ = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                             logit_chunk=8)
        ones, _, _ = trainer.masked_forward(model, ids, lbl, need_logits=False,
                                            logit_chunk=8, weights=torch.ones(VOCAB))
    assert torch.allclose(plain, ones, atol=1e-6)
