"""Offline top-K teacher logits for knowledge distillation (architecture-agnostic).

Why offline: the in-loop KD path (``train.py --kd-teacher``) holds a dense teacher in memory
*alongside* the student, which does not fit when the student is already training all 36 layers
at the memory ceiling. Running the teacher once over the corpus and storing only the top-K
logprobs per labeled position costs ~0.4 KB/position (measured: 125 MB for a 217-window corpus,
vs ~16 GB resident) and makes KD training CHEAPER than plain CE, because the teacher never runs
during training at all.

Measured on SWE-Lego-Qwen3-8B over the iter-5 verified-trajectory corpus: top-64 captures
99.8% of the teacher's probability mass (median 100%), the teacher's top-1 matches the gold
label 83.7% of the time, and the gold label falls inside the stored top-64 at 98.9% of
positions — i.e. the truncation keeps essentially all of the usable KD signal.

Storage: one ``.pt`` per corpus with a flat table over labeled positions

    win     int32  [P]      which corpus window
    pos     int32  [P]      which position within that window (student-shift semantics)
    idx     int32  [P, K]   teacher's top-K token ids
    logp    float16[P, K]   teacher's log-softmax values at those ids (already temperature 1)
    tail    float16[P]      log of the probability mass OUTSIDE the top-K (for exact renorm)

Architecture flexibility (this is deliberately not Qwen-specific):
  * ``resolve_vocab_size`` walks nested configs (``text_config`` / ``llm_config`` / ``decoder``),
    so multimodal wrappers (e.g. ``BeeForConditionalGeneration``) resolve to their LM vocab.
  * ``load_teacher`` tolerates real-world config quirks seen in the wild, e.g. a float
    ``max_position_embeddings`` (163840.0) that strict validators reject.
  * ``tokenizer_compatibility`` compares actual id->token STRINGS, not just ``vocab_size``.
    A padded embedding matrix (teacher config 151936 vs student 151669, identical tokenizers)
    is the common case and is safe: we slice logits to the shared prefix. Genuinely different
    tokenizers are refused, loudly, because per-token KD would be silently wrong.
  * The teacher forward uses ``logits_to_keep`` when the architecture supports it and falls
    back to a full-logits gather when it does not.
  * Top-K storage is vocab-size agnostic (ids + values), so moving to a larger-vocab model
    later needs no format change.

    .venv/bin/python scripts/kd_precompute.py --teacher SWE-Lego/SWE-Lego-Qwen3-8B \\
        --corpus out/exp-058/distill_corpus_iter5-r3.pt --max-windows 4 --out out/exp-058/kd_topk_smoke.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from quant_tuner.qat._device import resolve_backend

DEFAULT_TOPK = 64
POS_CHUNK = 4096  # positions per fp32 softmax chunk (~2.5 GiB transient at 151k vocab)


# --------------------------------------------------------------------------- config helpers
def resolve_vocab_size(config: Any) -> int:
    """Vocab size of the language head, walking nested configs.

    Multimodal/composite architectures put the LM config under ``text_config`` (Bee, Llava),
    ``llm_config``, or ``decoder``. Returns the first ``vocab_size`` found.
    """
    if getattr(config, "vocab_size", None):
        return int(config.vocab_size)
    for attr in ("text_config", "llm_config", "language_model_config", "decoder"):
        sub = getattr(config, attr, None)
        if sub is not None:
            try:
                return resolve_vocab_size(sub)
            except ValueError:
                continue
    raise ValueError(f"could not resolve vocab_size from config {type(config).__name__}")


def _sanitize_config_dict(cfg: dict) -> dict:
    """Coerce known-bad field types some published configs ship with.

    Seen in the wild: ``max_position_embeddings: 163840.0`` (float) which transformers'
    strict dataclass validation rejects with a TypeError.
    """
    int_fields = ("max_position_embeddings", "vocab_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "hidden_size",
                  "intermediate_size", "head_dim", "sliding_window")
    out = dict(cfg)
    for f in int_fields:
        v = out.get(f)
        if isinstance(v, float) and float(v).is_integer():
            out[f] = int(v)
    for sub in ("text_config", "llm_config", "decoder"):
        if isinstance(out.get(sub), dict):
            out[sub] = _sanitize_config_dict(out[sub])
    return out


def load_teacher(teacher: str | Path, *, device: str, dtype: torch.dtype):
    """Load any HF causal LM as a frozen teacher, tolerating config quirks."""
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        cfg = AutoConfig.from_pretrained(teacher)
    except Exception as e:  # e.g. float max_position_embeddings -> strict validation error
        print(f"[kd] config load failed ({type(e).__name__}); retrying with sanitized fields",
              flush=True)
        from huggingface_hub import hf_hub_download
        src = Path(teacher) / "config.json"
        raw = json.loads(src.read_text()) if src.exists() else json.loads(
            Path(hf_hub_download(str(teacher), "config.json")).read_text())
        cfg = AutoConfig.for_model(**_sanitize_config_dict(raw))

    model = AutoModelForCausalLM.from_pretrained(teacher, config=cfg, dtype=dtype)
    model.to(device).eval().requires_grad_(False)
    return model


def load_tokenizer_tolerant(name: str | Path):
    """Load a tokenizer even when the repo's model config trips strict validation.

    ``AutoTokenizer.from_pretrained`` pulls the model config first; a malformed config kills
    it. Fall back to fetching only the tokenizer files into a temp dir.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        import shutil
        import tempfile

        from huggingface_hub import hf_hub_download
        tmp = Path(tempfile.mkdtemp())
        got = False
        for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json"):
            try:
                shutil.copy(hf_hub_download(str(name), fn), tmp / fn)
                got = True
            except Exception:
                continue
        if not got:
            raise
        return AutoTokenizer.from_pretrained(tmp)


# ----------------------------------------------------------------------- vocab compatibility
def tokenizer_compatibility(student_tok, teacher_tok) -> tuple[int, str]:
    """Return (n_shared_ids, human report), raising if the tokenizers disagree.

    KD is only meaningful when a token id means the SAME string to both models. A teacher whose
    ``config.vocab_size`` exceeds the student's is fine when the extra rows are embedding
    padding (identical tokenizers) — we slice to the shared prefix. Divergent id->string maps
    are refused: per-token KL across different tokenizers is silently wrong, not approximate.
    """
    sv, tv = student_tok.get_vocab(), teacher_tok.get_vocab()
    inv_s = {i: t for t, i in sv.items()}
    inv_t = {i: t for t, i in tv.items()}
    n = min(max(inv_s) + 1, max(inv_t) + 1)
    shared = [i for i in range(n) if i in inv_s and i in inv_t]
    mismatched = [i for i in shared if inv_s[i] != inv_t[i]]
    if mismatched:
        ex = [(i, inv_s[i], inv_t[i]) for i in mismatched[:5]]
        raise ValueError(
            f"teacher/student tokenizers disagree on {len(mismatched)} of {len(shared)} shared "
            f"ids (e.g. {ex}). Per-token KD needs a shared tokenizer; use trajectory-level "
            f"(data) distillation instead.")
    report = (f"tokenizers agree on all {len(shared)} shared ids "
              f"(student {len(sv)}, teacher {len(tv)}); KD vocab = {len(shared)}")
    return len(shared), report


# ------------------------------------------------------------------------------- precompute
def force_into_support(
    ids_k: torch.Tensor, vals: torch.Tensor, logp_full: torch.Tensor,
    include_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Guarantee ``include_ids`` appear in every stored support row.

    The tail bucket in ``kd_loss_from_topk`` caps the student's TOTAL out-of-support
    mass, but says nothing about which token carries it. Forcing the stop token into
    the support makes the KL an exact per-position constraint on P(stop) — the j-th
    forced id replaces the (last-j)-th slot (the lowest-prob entries) in rows where it
    is absent, at its TRUE teacher logprob, and the tail is recomputed. Rows that
    already contain the id are untouched. Support rows are no longer sorted by prob
    afterwards; nothing downstream relies on that.
    """
    if len(include_ids) > ids_k.shape[-1]:
        raise ValueError(f"{len(include_ids)} forced ids > top-{ids_k.shape[-1]} support")
    for j, fid in enumerate(include_ids):
        absent = ~(ids_k == fid).any(-1)
        if absent.any():
            slot = ids_k.shape[-1] - 1 - j
            ids_k[absent, slot] = fid
            vals[absent, slot] = logp_full[absent, fid]
    tail = torch.log1p(-torch.exp(vals).sum(-1).clamp(max=1 - 1e-6))
    return ids_k, vals, tail


def _teacher_logits(teacher, ids: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """[K, V] teacher logits at ``keep_idx``, using logits_to_keep when supported.

    Returned in the model's own dtype — the fp32 upcast happens per position chunk in
    ``precompute_topk``. A 32B teacher's [20k, 151k] logits are 11.5 GiB in fp32; two
    whole-tensor fp32 copies (upcast + log_softmax) OOM'd a 95 GiB card that holds the
    bf16 weights (65 GiB) with room for exactly one of them.
    """
    try:
        out = teacher(input_ids=ids, logits_to_keep=keep_idx)
        lg = out.logits
        if lg.shape[1] == keep_idx.numel():          # honored the index tensor
            return lg[0]
        return lg[0].index_select(0, keep_idx)       # returned full seq anyway
    except TypeError:                                 # architecture lacks the kwarg
        return teacher(input_ids=ids).logits[0].index_select(0, keep_idx)


def _topk_rows(
    lg: torch.Tensor, topk: int, include_ids: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K support rows (logp, ids, tail) from [K, V] logits in any dtype.

    The fp32 softmax/top-K runs in POS_CHUNK-position chunks: peak transient is one
    [POS_CHUNK, V] fp32 pair instead of two whole-[K, V] copies. Chunking is exact —
    log_softmax, topk and force_into_support are all rowwise.
    """
    v_l, i_l, t_l = [], [], []
    for i in range(0, lg.shape[0], POS_CHUNK):
        logp_full = torch.log_softmax(lg[i:i + POS_CHUNK].float(), dim=-1)
        vals_c, ids_c = torch.topk(logp_full, topk, dim=-1)
        if include_ids:
            ids_c, vals_c, tail_c = force_into_support(ids_c, vals_c, logp_full,
                                                       include_ids)
        else:
            # exact tail mass so training can renormalize without bias
            tail_c = torch.log1p(-torch.exp(vals_c).sum(-1).clamp(max=1 - 1e-6))
        v_l.append(vals_c)
        i_l.append(ids_c)
        t_l.append(tail_c)
        del logp_full
    return torch.cat(v_l), torch.cat(i_l), torch.cat(t_l)


def precompute_topk(
    *,
    corpus: Path,
    teacher: str | Path,
    out: Path,
    student_model_dir: Path,
    topk: int = DEFAULT_TOPK,
    max_windows: int | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    student_chat_template: Path | None = None,
    include_ids: list[int] | None = None,
) -> dict:
    """Run the teacher over ``corpus`` and store top-K logprobs at labeled positions.

    ``include_ids`` forces those token ids into every stored support row (see
    :func:`force_into_support`) — pass the stop id so the KL constrains P(stop)
    per-position instead of only through the tail bucket.
    """
    backend = resolve_backend(device or "auto")
    dev = backend.name
    tdtype = dtype or backend.teacher_dtype

    blob = torch.load(corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n_win = ids_all.shape[0] if max_windows is None else min(max_windows, ids_all.shape[0])
    corpus_fp = blob.get("fingerprint")

    # --- vocab / tokenizer compatibility BEFORE loading 16 GB of weights -------------------
    from transformers import AutoConfig
    stok = load_tokenizer_tolerant(student_model_dir)
    if student_chat_template and Path(student_chat_template).exists():
        stok.chat_template = Path(student_chat_template).read_text()
    ttok = load_tokenizer_tolerant(teacher)
    n_shared, report = tokenizer_compatibility(stok, ttok)
    print(f"[kd] {report}", flush=True)

    s_vocab = resolve_vocab_size(AutoConfig.from_pretrained(student_model_dir))
    kd_vocab = min(n_shared, s_vocab)
    if topk > kd_vocab:
        topk = kd_vocab
    print(f"[kd] student vocab {s_vocab}; KD restricted to first {kd_vocab} ids; top-K={topk}",
          flush=True)

    print(f"[kd] loading teacher {teacher} ({tdtype}, {dev}) ...", flush=True)
    model = load_teacher(teacher, device=dev, dtype=tdtype)
    t_vocab = resolve_vocab_size(model.config)
    print(f"[kd] teacher vocab {t_vocab} (slicing logits to {kd_vocab})", flush=True)

    wins: list[torch.Tensor] = []
    poss: list[torch.Tensor] = []
    idxs: list[torch.Tensor] = []
    logps: list[torch.Tensor] = []
    tails: list[torch.Tensor] = []
    t0 = time.time()
    n_pos_total = 0
    for w in range(n_win):
        lbl = lbl_all[w:w + 1]
        tgt = lbl[:, 1:]
        keep = (tgt[0] != -100).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        ids = ids_all[w:w + 1].to(dev)
        keep_dev = keep.to(dev)
        with torch.no_grad():
            lg = _teacher_logits(model, ids, keep_dev)[:, :kd_vocab]
            vals, ids_k, tail = _topk_rows(lg, topk, include_ids)
        wins.append(torch.full((keep.numel(),), w, dtype=torch.int32))
        poss.append(keep.to(torch.int32))
        idxs.append(ids_k.to(torch.int32).cpu())
        logps.append(vals.to(torch.float16).cpu())
        tails.append(tail.to(torch.float16).cpu())
        n_pos_total += keep.numel()
        if (w + 1) % 5 == 0 or w == n_win - 1:
            el = time.time() - t0
            print(f"[kd] window {w+1}/{n_win}  positions={n_pos_total}  "
                  f"{el:.0f}s ({el/(w+1):.1f}s/window)", flush=True)
        del lg
        backend.empty_cache()

    payload = {
        "win": torch.cat(wins), "pos": torch.cat(poss),
        "idx": torch.cat(idxs), "logp": torch.cat(logps), "tail": torch.cat(tails),
        "topk": topk, "kd_vocab": kd_vocab,
        "teacher": str(teacher), "teacher_vocab": t_vocab, "student_vocab": s_vocab,
        "corpus": str(corpus), "corpus_fingerprint": corpus_fp,
        "n_windows": n_win, "n_positions": n_pos_total,
        "include_ids": list(include_ids or []),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    mb = out.stat().st_size / 1024**2
    print(f"[kd] saved {n_pos_total} positions x top-{topk} -> {out} ({mb:.1f} MB; "
          f"{mb * 1024 / max(1, n_pos_total):.2f} KB/position)", flush=True)
    return payload


def kd_loss_from_topk(
    student_logits: torch.Tensor, idx: torch.Tensor, logp: torch.Tensor,
    tail: torch.Tensor | None = None, temp: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) over the stored top-K support plus a tail bucket.

    ``student_logits`` [K_pos, V]; ``idx``/``logp`` [K_pos, K]; ``tail`` [K_pos] is the
    teacher's log-mass outside its top-K.

    With ``tail`` given (and ``temp == 1``) the KL is taken over K+1 buckets: the K stored
    tokens at their TRUE probabilities plus one bucket for everything else (teacher side
    stored at precompute, student side ``1 - Σ support``). This term is the whole point:
    renormalizing both sides over the top-K — the previous form — makes the loss blind to
    any student mass placed OUTSIDE the teacher's support (inflating an out-of-support
    logit deflates every support prob proportionally, and renormalization cancels it
    exactly). Measured on our corpus, ``<|im_end|>`` is outside the teacher's top-64 at
    98.2% of supervised positions, so the termination collapse — P(stop) rising exactly
    where the teacher keeps it in the tail — was invisible to the renormalized KL. The
    tail bucket caps the student's total out-of-support mass at the teacher's (~0.006
    mean), which pins P(stop) as a side effect. An identical student still scores 0.

    With ``temp != 1`` the tail bucket is skipped (a T-tempered tail is not derivable from
    the stored T=1 tail) and both sides fall back to top-K renormalization.

    Returns a scalar averaged over positions.
    """
    s_logp = torch.log_softmax(student_logits / temp, dim=-1)
    s_at = s_logp.gather(-1, idx.long())
    t_logp = logp.float() / temp
    if tail is not None and temp == 1.0:
        t_sup = t_logp.exp()
        t_tail = tail.float().exp().clamp_min(1e-8)
        s_tail = (1.0 - s_at.exp().sum(-1)).clamp_min(1e-8)
        kl = ((t_sup * (t_logp - s_at)).sum(-1)
              + t_tail * (t_tail.log() - s_tail.log()))
        return kl.mean()
    s_at = s_at - torch.logsumexp(s_at, dim=-1, keepdim=True)       # renormalize over top-K
    t_logp = t_logp - torch.logsumexp(t_logp, dim=-1, keepdim=True)  # renormalize over top-K
    t_p = t_logp.exp()
    return (t_p * (t_logp - s_at)).sum(-1).mean()
