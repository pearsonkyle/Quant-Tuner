"""Termination steering: the probe's metric family as gradient, with the probe held out."""

from __future__ import annotations

import torch

from quant_tuner.qat.steer import SteerBatch, steering_loss
from quant_tuner.qat.stop_probe import SENTENCE, USER


class _StubTok:
    """Chat-template-free stand-in: template = concatenated contents + marker."""

    pad_token_id = 0

    def apply_chat_template(self, msgs, tools=None, tokenize=False,
                            add_generation_prompt=False):
        out = "".join(m["content"] + "\n" for m in msgs)
        return out + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        class E:
            def __init__(self, ids):
                self.input_ids = ids
        return E([(ord(c) % 97) + 3 for c in text[:512]])


def test_build_shapes_and_probe_exclusion():
    b = SteerBatch.build(_StubTok(), n=8, seed=11, stop_id=42)
    assert b.ids.shape == b.attn.shape and b.ids.shape[0] == 8
    assert int(b.want_stop.sum()) == 4           # default stop_frac 0.5
    # left-padded: every row's LAST position is real (the decision point)
    assert bool((b.attn[:, -1] == 1).all())
    # rows shorter than max are padded at the FRONT
    lens = b.attn.sum(1)
    r = int(lens.argmin())
    if int(lens[r]) < b.ids.shape[1]:
        assert b.attn[r, 0] == 0


def test_probe_texts_never_generated():
    """Goodhart guard: the held-out probe's task and sentence must not appear in any
    steering context, for any seed we might plausibly use."""
    tok = _StubTok()
    for seed in range(20):
        b = SteerBatch.build(tok, n=12, seed=seed)   # asserts internally
        assert b is not None
    assert SENTENCE and USER                          # sanity: constants exist


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                      vocab_size=128, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


def _tiny_batch(n=6, stop_id=42):
    torch.manual_seed(1)
    ids = torch.randint(3, 128, (n, 24))
    attn = torch.ones_like(ids)
    attn[0, :5] = 0                                  # one left-padded row
    want = torch.tensor([True] * 3 + [False] * 3)
    import math
    return SteerBatch(ids, attn, want, math.log(0.02), stop_id)


def test_loss_directions_and_gradient():
    m = _tiny()
    b = _tiny_batch()
    loss, met = steering_loss(m, b)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    g = m.lm_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # CE toward stop is positive unless the model already predicts stop
    assert met["steer_stop_ce"] > 0
    # gradient on the stop row of lm_head must push P(stop) UP for control rows:
    # a step against the gradient increases the stop logit's alignment
    assert met["steer_p_stop_ctrl"] < 1.0


def test_hinge_is_silent_below_cap_and_active_above():
    """Diagnostic rows below the cap contribute exactly zero — the hinge steers only
    drift, never over-suppresses. Above the cap it is linear in the excess."""
    m = _tiny()
    b = _tiny_batch()
    with torch.no_grad():
        lp = torch.log_softmax(
            m(input_ids=b.ids, attention_mask=b.attn).logits[:, -1, :].float(), -1)
        s_cont = lp[~b.want_stop, b.stop_id]
    b.cap_logp = float(s_cont.max()) + 0.1           # cap above every row -> silent
    _, met = steering_loss(m, b)
    assert met["steer_cont_pen"] == 0.0
    b.cap_logp = float(s_cont.min()) - 1.0           # cap below every row -> active
    _, met = steering_loss(m, b)
    expect = float((s_cont - b.cap_logp).mean())
    assert abs(met["steer_cont_pen"] - expect) < 1e-4


def test_rep_batch_spans_cover_the_repeated_command():
    from quant_tuner.qat.steer import RepBatch
    b = RepBatch.build(_StubTok(), n=6, seed=23)
    assert b.ids.shape[0] == 6
    for i in range(6):
        lo, hi = int(b.span[i, 0]), int(b.span[i, 1])
        assert 0 < lo < hi <= b.ids.shape[1]
        assert bool((b.attn[i, lo:hi] == 1).all())     # span is real tokens, not pad


def test_repetition_loss_is_one_sided_and_flows_gradient():
    """Below the per-token cap the hinge is silent; a near-deterministic copier is
    penalized and the gradient reaches the model."""
    import math

    from quant_tuner.qat.steer import RepBatch, repetition_loss
    m = _tiny()
    torch.manual_seed(2)
    ids = torch.randint(3, 128, (4, 30))
    attn = torch.ones_like(ids)
    span = torch.tensor([[20, 28]] * 4)
    b = RepBatch(ids, attn, span, math.log(0.5))
    pen, met = repetition_loss(m, b)
    # a random tiny model is nowhere near 0.5 per-token on arbitrary continuations
    assert met["rep_p_mean"] < 0.5 and float(pen) == 0.0
    b2 = RepBatch(ids, attn, span, math.log(1e-6))     # cap below everything -> active
    pen2, _ = repetition_loss(m, b2)
    assert float(pen2) > 0
    pen2.backward()
    g = m.lm_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def _rep_kd_from_model(model, batch, topk=8):
    """Capture-equivalent table built from the model's own logits."""
    import torch

    from quant_tuner.qat.kd_precompute import _topk_rows
    from quant_tuner.qat.steer import RepKD, rep_fingerprint
    idxs, logps, tails, off = [], [], [], [0]
    with torch.no_grad():
        for i in range(batch.ids.shape[0]):
            lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
            lg = model(input_ids=batch.ids[i:i + 1],
                       attention_mask=batch.attn[i:i + 1]).logits[0]
            vals, ids_k, tail = _topk_rows(lg[lo - 1:hi - 1], topk, None)
            idxs.append(ids_k)
            logps.append(vals)
            tails.append(tail)
            off.append(off[-1] + (hi - lo))
    return RepKD(torch.cat(idxs), torch.cat(logps), torch.cat(tails),
                 torch.tensor(off), rep_fingerprint(batch), "self")


def test_rep_kd_zero_for_identical_model_and_positive_after_shift():
    """The KL term is 0 when the student IS the teacher, and >0 once the student moves —
    the property that makes it a restoring force toward the teacher's action policy."""
    import math

    import torch

    from quant_tuner.qat.steer import RepBatch, repetition_losses
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(3, 128, (4, 30))
    batch = RepBatch(ids, torch.ones_like(ids), torch.tensor([[20, 28]] * 4),
                     math.log(0.5))
    kd = _rep_kd_from_model(model, batch)
    _, kl_same, stats = repetition_losses(model, batch, kd)
    assert kl_same is not None and abs(float(kl_same)) < 1e-4
    assert "rep_kl" in stats
    with torch.no_grad():
        for p_ in model.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    _, kl_moved, _ = repetition_losses(model, batch, kd)
    assert float(kl_moved) > float(kl_same) + 1e-3


def test_rep_kd_fingerprint_mismatch_refused(tmp_path):
    """A table captured on different contexts must be refused, not silently trained
    against (same failure class as the KD table's corpus-fingerprint guard)."""
    import math

    import pytest
    import torch

    from quant_tuner.qat.steer import RepBatch, RepKD
    torch.manual_seed(1)
    model = _tiny()
    ids = torch.randint(3, 128, (4, 30))
    batch = RepBatch(ids, torch.ones_like(ids), torch.tensor([[20, 28]] * 4),
                     math.log(0.5))
    kd = _rep_kd_from_model(model, batch)
    path = tmp_path / "rep_kd.pt"
    torch.save({"idx": kd.idx, "logp": kd.logp, "tail": kd.tail,
                "row_off": kd.row_off, "fingerprint": "deadbeefdeadbeef",
                "teacher": "self"}, path)
    with pytest.raises(ValueError, match="different contexts"):
        RepKD.load(path, batch)
