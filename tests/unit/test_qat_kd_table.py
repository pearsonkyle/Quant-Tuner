"""Offline KD: the table lookup and the chunked loss.

Every failure pinned here is silent. A KD table resolves against ANY corpus with enough
windows, and a mismatched one distils each student position against some other token's
teacher distribution while the loss curve looks entirely normal — so the guards are the
feature, not the plumbing.
"""

from __future__ import annotations

import pytest
import torch

from quant_tuner.qat.kd_precompute import kd_loss_from_topk
from quant_tuner.qat.kd_table import KDTable


def _payload(n_win=3, per_win=4, topk=5, vocab=50, fp="abc123", seed=0):
    g = torch.Generator().manual_seed(seed)
    win, pos, idx, logp, tail = [], [], [], [], []
    for w in range(n_win):
        for j in range(per_win):
            win.append(w)
            pos.append(j * 2)                      # strictly increasing within a window
            idx.append(torch.randperm(vocab, generator=g)[:topk])
            lp = torch.log_softmax(torch.randn(topk, generator=g), dim=-1)
            logp.append(lp)
            tail.append(torch.tensor(-5.0))
    return {
        "win": torch.tensor(win, dtype=torch.int32),
        "pos": torch.tensor(pos, dtype=torch.int32),
        "idx": torch.stack(idx).to(torch.int32),
        "logp": torch.stack(logp).to(torch.float16),
        "tail": torch.stack(tail).to(torch.float16),
        "topk": topk, "kd_vocab": vocab, "teacher": "fake/teacher",
        "corpus_fingerprint": fp, "n_windows": n_win, "n_positions": n_win * per_win,
    }


# ------------------------------------------------------------------ table integrity
def test_fingerprint_mismatch_is_refused():
    """A table built from another pack resolves fine — same window count, same position
    range — and would distil every position against the wrong distribution."""
    with pytest.raises(ValueError, match="wrong teacher distribution|built from corpus"):
        KDTable(_payload(fp="AAA"), corpus_fingerprint="BBB")


def test_matching_fingerprint_is_accepted():
    t = KDTable(_payload(fp="same"), corpus_fingerprint="same")
    assert t.n_windows == 3 and t.topk == 5


def test_missing_fingerprint_on_either_side_does_not_block():
    """Older tables predate the field; refusing them would be a false alarm."""
    p = _payload()
    p["corpus_fingerprint"] = None
    KDTable(p, corpus_fingerprint="whatever")
    KDTable(_payload(fp="x"), corpus_fingerprint=None)


def test_window_rows_are_sliced_not_scanned():
    t = KDTable(_payload(n_win=3, per_win=4))
    keep = torch.tensor([0, 2, 4, 6])
    w1 = t.for_window(1, keep)
    assert len(w1) == 4
    # window 1's rows must be its own, not window 0's
    torch.testing.assert_close(w1.logp.float(),
                               t._logp[4:8].float())


def test_position_mismatch_is_refused_even_at_the_same_count():
    """The dangerous case: the trainer selected as many positions as the table stored, but
    different ones. Pairing them would misalign every single row."""
    t = KDTable(_payload(per_win=4))
    with pytest.raises(ValueError, match="same count, different positions"):
        t.for_window(0, torch.tensor([1, 3, 5, 7]))


def test_position_count_mismatch_is_refused():
    t = KDTable(_payload(per_win=4))
    with pytest.raises(ValueError, match="supervised positions"):
        t.for_window(0, torch.tensor([0, 2]))


def test_empty_window_raises_rather_than_silently_skipping_kd():
    p = _payload(n_win=3, per_win=2)
    keep = p["win"] != 1                       # window 1 has no rows
    for k in ("win", "pos", "idx", "logp", "tail"):
        p[k] = p[k][keep]
    t = KDTable(p)
    assert not t.has_window(1)
    with pytest.raises(KeyError):
        t.for_window(1, torch.tensor([0, 2]))


def test_coverage_reports_mass_outside_the_topk():
    p = _payload()
    p["tail"] = torch.full((p["tail"].shape[0],), -0.6931).to(torch.float16)  # log(0.5)
    assert KDTable(p).coverage() == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------- forced support ids
def test_force_into_support_inserts_absent_id_at_true_logprob():
    from quant_tuner.qat.kd_precompute import force_into_support
    torch.manual_seed(5)
    vocab, K, topk = 40, 6, 5
    logp_full = torch.log_softmax(torch.randn(K, vocab), dim=-1)
    vals, ids_k = torch.topk(logp_full, topk, dim=-1)
    fid = int(logp_full.mean(0).argmin())          # a low-prob id, absent from most rows
    before = (ids_k == fid).any(-1).clone()
    ids_k, vals, tail = force_into_support(ids_k, vals, logp_full, [fid])
    assert (ids_k == fid).any(-1).all()
    # inserted at the teacher's TRUE logprob, and the tail matches the new support
    for r in range(K):
        j = (ids_k[r] == fid).nonzero()[0, 0]
        torch.testing.assert_close(vals[r, j], logp_full[r, fid])
        torch.testing.assert_close(
            tail[r], torch.log1p(-vals[r].exp().sum().clamp(max=1 - 1e-6)))
    # rows that already had the id keep their full original support
    for r in before.nonzero(as_tuple=True)[0]:
        assert set(ids_k[r].tolist()) == set(torch.topk(logp_full[r], topk).indices.tolist())


def test_force_into_support_refuses_more_ids_than_slots():
    from quant_tuner.qat.kd_precompute import force_into_support
    logp_full = torch.log_softmax(torch.randn(2, 10), dim=-1)
    vals, ids_k = torch.topk(logp_full, 2, dim=-1)
    with pytest.raises(ValueError, match="forced ids"):
        force_into_support(ids_k, vals, logp_full, [0, 1, 2])


# ------------------------------------------------------------------------ the loss
def test_identical_student_scores_zero():
    """Both sides must be renormalized over the stored top-K. Normalizing only the teacher
    leaves the student's missing tail mass as a constant offset, and an identical student
    scored 0.89 instead of 0."""
    torch.manual_seed(1)
    vocab, K, topk = 40, 6, 5
    logits = torch.randn(K, vocab)
    lp = torch.log_softmax(logits, dim=-1)
    val, idx = torch.topk(lp, topk, dim=-1)
    tail = torch.log1p(-val.exp().sum(-1).clamp(max=1 - 1e-6))
    kl = kd_loss_from_topk(logits, idx.to(torch.int32), val.to(torch.float16), tail)
    assert abs(float(kl)) < 1e-3


def test_tail_bucket_sees_out_of_support_mass():
    """The regression the renormalized form missed: a student that moves mass onto a token
    OUTSIDE the teacher's top-K — the termination collapse, since <|im_end|> is outside the
    teacher's top-64 at 98.2% of our positions — must be penalized. Renormalizing both
    sides over the support cancels an out-of-support logit exactly (it shifts every support
    logprob by the same logsumexp constant), so the old form scores this student 0."""
    torch.manual_seed(4)
    vocab, K, topk = 40, 6, 5
    t_logits = torch.randn(K, vocab)
    lp = torch.log_softmax(t_logits, dim=-1)
    val, idx = torch.topk(lp, topk, dim=-1)
    tail = torch.log1p(-val.exp().sum(-1).clamp(max=1 - 1e-6))
    bad = t_logits.clone()
    bad[torch.arange(K), lp.argmin(-1)] += 8.0   # inflate a token never in the top-5
    idx32, val16 = idx.to(torch.int32), val.to(torch.float16)
    with_tail = kd_loss_from_topk(bad, idx32, val16, tail)
    blind = kd_loss_from_topk(bad, idx32, val16, None)
    assert float(with_tail) > 0.5, f"tail bucket must see the drift, got {float(with_tail)}"
    assert abs(float(blind)) < 1e-4, "support-renormalized KL is blind to it by construction"


def test_chunked_kd_matches_unchunked():
    """The trainer computes KD inside the CE logit chunks; a mean of per-chunk means would
    be wrong whenever K is not a multiple of the chunk, so the chunked path sums and
    divides by the true K."""
    torch.manual_seed(2)
    vocab, K, topk, chunk = 40, 7, 5, 3        # 7 is deliberately not a multiple of 3
    logits = torch.randn(K, vocab)
    t_logits = torch.randn(K, vocab)
    lp = torch.log_softmax(t_logits, dim=-1)
    val, idx = torch.topk(lp, topk, dim=-1)
    idx32, val16 = idx.to(torch.int32), val.to(torch.float16)
    tail = torch.log1p(-val.exp().sum(-1).clamp(max=1 - 1e-6)).to(torch.float16)

    whole = kd_loss_from_topk(logits, idx32, val16, tail)
    total = torch.zeros(())
    for i in range(0, K, chunk):
        n = logits[i:i + chunk].shape[0]
        total = total + kd_loss_from_topk(
            logits[i:i + chunk], idx32[i:i + chunk], val16[i:i + chunk],
            tail[i:i + chunk]) * n
    torch.testing.assert_close(total / K, whole, rtol=1e-5, atol=1e-6)


def test_kd_loss_is_positive_for_a_diverging_student():
    torch.manual_seed(3)
    vocab, K, topk = 40, 5, 5
    t_logits = torch.randn(K, vocab)
    lp = torch.log_softmax(t_logits, dim=-1)
    val, idx = torch.topk(lp, topk, dim=-1)
    tail = torch.log1p(-val.exp().sum(-1).clamp(max=1 - 1e-6))
    far = torch.randn(K, vocab) * 5
    assert float(kd_loss_from_topk(far, idx.to(torch.int32),
                                   val.to(torch.float16), tail)) > 0.1


def test_kd_window_slice_keeps_rows_aligned():
    t = KDTable(_payload(per_win=6, topk=5))
    w = t.for_window(0, torch.tensor([0, 2, 4, 6, 8, 10]))
    s = w.slice(2, 5)
    assert len(s) == 3
    torch.testing.assert_close(s.logp.float(), w.logp[2:5].float())


# ------------------------------------------------- masked_forward's KD integration
def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                      vocab_size=128, max_position_embeddings=256,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


def _kd_window_for(model, ids, lbl, topk=8):
    """Teacher rows from the model's OWN logits — so KL must be ~0 against itself."""
    from quant_tuner.qat.kd_table import KDWindow
    tgt = lbl[:, 1:]
    keep = (tgt[0] != -100).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[:, keep, :]
        lp = torch.log_softmax(model.lm_head(h)[0].float(), dim=-1)
    val, idx = torch.topk(lp, topk, dim=-1)
    tail = torch.log1p(-val.exp().sum(-1).clamp(max=1 - 1e-6))
    return KDWindow(idx.to(torch.int32), val.to(torch.float16), tail.to(torch.float16)), keep


def test_masked_forward_kd_chunked_equals_unchunked():
    """The chunked path is what actually runs (K x 151669 fp32 is 5.8 GiB at our density),
    so it has to agree with the whole-tensor computation it replaces."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100                       # 27 supervised targets after the shift
    kd, _ = _kd_window_for(m, ids, lbl)

    ce_w, _, _, kl_w, _ = masked_forward(m, ids, lbl, need_logits=False,
                                         logit_chunk=10_000, kd=kd)
    ce_c, _, _, kl_c, _ = masked_forward(m, ids, lbl, need_logits=False,
                                         logit_chunk=7, kd=kd)   # 27 % 7 != 0 on purpose
    torch.testing.assert_close(ce_c, ce_w, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(kl_c, kl_w, rtol=1e-4, atol=1e-6)


def test_masked_forward_kd_against_itself_is_near_zero():
    """Teacher == student, so the KL must vanish. Catches a normalization or an
    off-by-one in the position alignment, both of which leave a finite floor."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100
    kd, _ = _kd_window_for(m, ids, lbl)
    _, _, _, kl, _ = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7, kd=kd)
    kl_v = float(kl.detach())
    assert kl_v < 1e-3, f"KL against itself should vanish, got {kl_v}"


def test_masked_forward_without_kd_keeps_the_three_tuple():
    """The non-KD return arity is load-bearing for every existing caller."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 24))
    lbl = ids.clone()
    lbl[:, :8] = -100
    assert len(masked_forward(m, ids, lbl, need_logits=False, logit_chunk=5)) == 3


def test_kd_row_count_mismatch_raises_in_masked_forward():
    """Length is asserted at the point of use: any future path that filters keep_idx
    without filtering the KD rows must fail loudly, not silently misalign chunks."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100
    kd, _ = _kd_window_for(m, ids, lbl)
    with pytest.raises(ValueError, match="wrong positions"):
        masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7, kd=kd.slice(0, 5))


def test_kd_rows_follow_prefix_filtering():
    """With n_prefix > 0 masked_forward drops targets inside the prefix; the KD window was
    validated against the FULL keep set, so its leading rows must be dropped in lockstep
    (before this fix they weren't, and every chunk paired a student position with an
    earlier token's teacher distribution)."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100                       # 27 supervised targets after the shift
    kd, keep = _kd_window_for(m, ids, lbl)   # rows for ALL 27, position order
    n_prefix = 20
    n_tail = int((keep >= n_prefix).sum())
    assert 0 < n_tail < len(kd)              # the prefix must actually drop some rows
    out = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7,
                         n_prefix=n_prefix, kd=kd)
    assert len(out) == 5 and torch.isfinite(out[3])


def test_kd_gradients_flow_to_the_student():
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 32))
    lbl = ids.clone()
    lbl[:, :10] = -100
    kd = _kd_window_for(m, ids, lbl)[0]
    # perturb so the KL is non-zero and has a gradient
    with torch.no_grad():
        m.lm_head.weight.mul_(1.4)
    _, _, _, kl, _ = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=6, kd=kd)
    kl.backward()
    g = m.lm_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ----------------------------------------------------------------- the stop anchor
def test_stop_logp_of_extracts_and_refuses_plain_tables():
    from quant_tuner.qat.kd_table import stop_logp_of
    p = _payload(n_win=1, per_win=4, topk=5, vocab=50)
    t = KDTable(p)
    w = t.for_window(0, torch.tensor([0, 2, 4, 6]))
    sid = int(w.idx[0, 2])                       # present in row 0, not guaranteed in all
    if bool((w.idx == sid).any(-1).all()):       # rare; force absence in one row
        w.idx[1] = torch.arange(5, dtype=torch.int32) + 40
    with pytest.raises(ValueError, match="lack the stop id"):
        stop_logp_of(w, sid)
    # forced table: put sid in every row's last slot
    w.idx[:, -1] = sid
    w.idx[:, :-1][w.idx[:, :-1] == sid] = 39     # keep exactly one hit per row
    got = stop_logp_of(w, sid)
    torch.testing.assert_close(got, w.logp.float()[w.idx == sid])


def test_stop_anchor_is_one_sided_per_position_type():
    """Continue-positions outnumber stop-positions 176:1, so a symmetric hinge exerts a
    net-DOWNWARD trunk pressure on P(stop) — measured: diagnostic pinned at 0.0000 while
    the control collapsed 0.9987 -> 0.6974. One-sided: at continue-positions (teacher
    P(stop) < 0.5) only stopping MORE than the teacher is penalized; at stop-positions
    only stopping LESS. Zero force on the safe side of each."""
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100
    kd, keep = _kd_window_for(m, ids, lbl, topk=8)
    sid = 5
    kd.idx[:, -1] = sid                           # force the "stop" id into support
    with torch.no_grad():
        h = m.model(input_ids=ids).last_hidden_state[:, keep, :]
        s_stop = torch.log_softmax(m.lm_head(h)[0].float(), -1)[:, sid]
    # tiny random model: s_stop is well below log(0.5), so these are continue-positions
    assert bool((s_stop < -0.694).all())
    cases = (
        (s_stop.clone(), 0.0),        # aligned -> 0
        (s_stop - 0.5, 0.0),          # student 0.5 nat ABOVE teacher, inside margin -> 0
        (s_stop - 3.0, 2.0),          # student 3 nats ABOVE teacher (stops too much) -> 2
        (s_stop + 3.0, 0.0),          # student BELOW teacher: safe side, NO down-drag
    )
    for t_stop, expect in cases:
        _, _, _, _, an = masked_forward(
            m, ids, lbl, need_logits=False, logit_chunk=7, kd=kd,
            stop_anchor=(sid, t_stop, 1.0))
        assert float(an) == pytest.approx(expect, abs=0.05), float(an)
    # stop-positions (teacher P(stop) > 0.5): only stopping LESS is penalized.
    t_hi = torch.full_like(s_stop, -0.05)         # teacher ~0.95 -> stop-position
    _, _, _, _, an = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7,
                                    kd=kd, stop_anchor=(sid, t_hi, 1.0))
    gap = float((-(s_stop - t_hi) - 1.0).clamp_min(0).mean())   # student far below
    assert float(an) == pytest.approx(gap, abs=0.05)


def test_stop_anchor_chunked_matches_unchunked_and_flows_gradient():
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100
    kd, keep = _kd_window_for(m, ids, lbl, topk=8)
    sid = 9
    kd.idx[:, -1] = sid
    t_stop = torch.full((int((lbl[:, 1:][0] != -100).sum()),), -8.0)
    _, _, _, _, a_c = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7,
                                     kd=kd, stop_anchor=(sid, t_stop, 1.0))
    _, _, _, _, a_w = masked_forward(m, ids, lbl, need_logits=False, logit_chunk=10_000,
                                     kd=kd, stop_anchor=(sid, t_stop, 1.0))
    torch.testing.assert_close(a_c, a_w, rtol=1e-4, atol=1e-6)
    a_c.backward()
    g = m.lm_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_stop_anchor_row_count_mismatch_raises():
    from quant_tuner.qat.train import masked_forward
    m = _tiny()
    ids = torch.randint(0, 128, (1, 40))
    lbl = ids.clone()
    lbl[:, :12] = -100
    kd, _ = _kd_window_for(m, ids, lbl)
    with pytest.raises(ValueError, match="anchor has"):
        masked_forward(m, ids, lbl, need_logits=False, logit_chunk=7, kd=kd,
                       stop_anchor=(3, torch.zeros(5), 1.0))
