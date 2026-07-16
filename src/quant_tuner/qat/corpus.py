"""QAT masked-corpus construction (log-based + strong-solver distillation).

Two corpora, one masking convention:

* :func:`build_log_corpus` — turn-aware, assistant-MASKED windows from the logtrain
  slices (the iter-2/3/4 corpus). Renders each session with the model's real chat
  template, masks loss to the ``<|im_start|>assistant … <|im_end|>`` spans INCLUDING the
  terminating ``<|im_end|>`` (the stop decision — without it no position ever trains
  end-of-turn, the mechanistic cause of the iter-2/3 looping), and packs to windows.
* :func:`build_distill_corpus` — the iter-5 data-quality lever: re-render a strong
  solver's VERIFIED (tests-pass) SWE-rebench trajectories through the student's tokenizer
  with the same masking. Keeps resolved-only by default (a patch that doesn't pass is the
  mimicry trap iter-4 fell into). Data/trajectory distillation, not logit-KD (the solver's
  vocab differs); same-vocab logit-KD is a composable trainer lever (:mod:`.train`).

Both share the masking primitives (:func:`masked_ids_for_session`, :func:`pack`,
:func:`corpus_fingerprint`) so the two corpora are bit-for-bit trainer-compatible and the
stop-token/density audits are identical. The CLI shims are ``scripts/build_qat_masked_corpus.py``
and ``scripts/build_ornith_distill_corpus.py``.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

from quant_tuner.data import split

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "out" / "exp-057" / "model"
CHAT_TEMPLATE = REPO / "out" / "exp-057" / "chat_template.jinja"
LOGTRAIN = REPO / "logtrain.jsonl"
WIKI = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"

# assistant span in Qwen render: from "<|im_start|>assistant\n" to the next "<|im_end|>"
_ASST_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)
IM_END = "<|im_end|>"

# The single bash tool the openai-agents SWE backend registers (kept in sync with
# eval/agents/openai_agents.py). Rendered into the student's # Tools block so the distill
# corpus trains schema-conditioned on the tool the model is actually given at eval.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository checkout at /testbed "
                       "and return its combined output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

# The SWE backend's system prompt (context, masked out; using the real one keeps the
# training distribution faithful to how the student is driven at eval).
SWE_SYSTEM_PROMPT = (
    "You are an autonomous software engineer fixing a real bug in a Python "
    "repository already checked out at /testbed (the current working directory).\n\n"
    "You have one tool: bash(command). Use it to explore the code, reproduce the "
    "issue, edit files, and run the project's tests. Make all edits directly on the "
    "files under /testbed with shell commands. Work in small steps and inspect "
    "command output before continuing.\n\n"
    "Your job is to FIX THE BUG IN THE LIBRARY/SOURCE CODE. Find the function or "
    "class responsible and change its implementation. First reproduce the failure, "
    "then edit the source, then re-run to confirm your source change makes the "
    "failing behavior pass. When the fix is complete and verified, stop and give a "
    "short summary of what you changed."
)


# --------------------------------------------------------------------------- masking core
def parse_session(s: dict) -> list[dict]:
    return [json.loads(m) if isinstance(m, str) else m for m in s["messages"]]


def reconstruct_tools(msgs: list[dict]) -> list[dict]:
    """FALLBACK ONLY: build per-session OpenAI-function schemas from the tools the
    session actually calls (name -> observed argument keys). The logtrain export
    stores the REAL schemas on the system message (``messages[0]["tools"]``) —
    resolved via ``split.session_tools`` — so this stub path should only fire for
    log formats that genuinely carry no schema block."""
    argk: dict[str, set] = collections.defaultdict(set)
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            name = fn.get("name") or tc.get("name")
            if not name:
                continue
            a = fn.get("arguments") or tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if isinstance(a, dict):
                argk[name] |= {k for k in a if isinstance(k, str) and k.isidentifier()}
    return [{"type": "function", "function": {
        "name": n, "description": f"{n} tool",
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string"} for k in sorted(ks)}}}}
            for n, ks in argk.items()]


def truncate_tool_messages(
    msgs: list[dict], tok, max_tool_tokens: int
) -> tuple[list[dict], int, int]:
    """Head+tail truncate ``role=="tool"`` string contents to ``max_tool_tokens``.

    Tool outputs carry their signal at both ends (command echo / final result +
    exit status); the middle of a 10k-token dump is masked context the model
    pays full forward/backward for. Returns (msgs, n_truncated, tokens_saved).
    Truncation happens BEFORE the chat-template render so spans/offsets stay
    consistent — they are computed on the final rendered text.
    """
    if max_tool_tokens <= 0:
        return msgs, 0, 0
    out: list[dict] = []
    n_trunc = 0
    saved = 0
    for m in msgs:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str):
            ids = tok(c, add_special_tokens=False)["input_ids"]
            if len(ids) > max_tool_tokens:
                head = tok.decode(ids[: max_tool_tokens // 2])
                tail = tok.decode(ids[-(max_tool_tokens - max_tool_tokens // 2):])
                dropped = len(ids) - max_tool_tokens
                m = {**m, "content": f"{head}\n…[truncated {dropped} tokens]…\n{tail}"}
                n_trunc += 1
                saved += dropped
        out.append(m)
    return out, n_trunc, saved


def masked_ids_for_session(
    msgs: list[dict], tok, tools: list | None = None
) -> tuple[list[int], list[int]]:
    """Return (ids, labels) with labels = ids on assistant tokens, -100 elsewhere.

    The labeled span runs from the first content character after the
    ``<|im_start|>assistant\\n`` header through the END of the terminating
    ``<|im_end|>`` (``m.end(0)``): the terminator is a trained target — it is the
    stop/EOS decision — while the header stays context (at inference it is part
    of the generation prompt, never generated). The token after ``<|im_end|>``
    (the turn-separating newline) stays -100.

    Tool schemas render in a system turn (# Tools block), so the assistant-span
    mask naturally excludes them — schemas are context, only the assistant's
    tool_calls/text (+ terminator) are trained."""
    if tools is None:
        tools = reconstruct_tools(msgs)
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                   add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # char spans: assistant *content* plus the terminating <|im_end|> (m.end(0))
    spans = [(m.start(1), m.end(0)) for m in _ASST_RE.finditer(text)]
    labels = [-100] * len(ids)
    si = 0
    for j, (a, b) in enumerate(offs):
        if a == b:  # special/zero-width
            continue
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and a >= spans[si][0] and b <= spans[si][1]:
            labels[j] = ids[j]
    return ids, labels


def pack(stream_ids: list[int], stream_lbl: list[int], window: int,
         min_trainable: int = 8, min_density: float = 0.0) -> list[dict]:
    """Chunk the stream into windows, DROPPING windows with < ``min_trainable``
    non-(-100) label tokens or below ``min_density`` trainable fraction. An
    all-masked window (a 4096 chunk landing entirely in a long tool output) has no
    gradient and yields a NaN CE loss (0/0); near-empty windows waste a full ~40s
    forward/backward on a few tokens of signal. ``min_density`` is the tunable
    wall-time lever — pick it from the density histogram the builders print."""
    n = (len(stream_ids) // window) * window
    out = []
    for i in range(0, n, window):
        lbl = stream_lbl[i:i + window]
        n_train = sum(1 for x in lbl if x != -100)
        if n_train >= min_trainable and n_train / window >= min_density:
            out.append({"ids": stream_ids[i:i + window], "labels": lbl})
    return out


def corpus_fingerprint(ids_t: torch.Tensor, lbl_t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(ids_t.shape)).encode())
    h.update(ids_t.numpy().tobytes())
    h.update(lbl_t.numpy().tobytes())
    return h.hexdigest()[:16]


def load_tokenizer(model_dir: Path = MODEL, chat_template: Path = CHAT_TEMPLATE):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.chat_template = Path(chat_template).read_text()
    return tok


def _finalize(windows: list[dict], window: int, tok, *, extra: dict, out: Path | None,
              tool_windows: int) -> dict:
    """Shuffle, tensorize, run the density + stop-token audit, save, return the blob."""
    rng = random.Random(42)
    rng.shuffle(windows)
    ids_t = torch.tensor([w["ids"] for w in windows], dtype=torch.long)
    lbl_t = torch.tensor([w["labels"] for w in windows], dtype=torch.long)

    dens = (lbl_t != -100).float().mean(dim=1)
    deciles = torch.quantile(dens, torch.linspace(0, 1, 11))
    print("[corpus] window trainable-density deciles: "
          + " ".join(f"{d:.2f}" for d in deciles.tolist()), flush=True)
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    n_im_end = (0 if im_end_id is None
                else int(((lbl_t == im_end_id) & (lbl_t != -100)).sum()))
    print(f"[corpus] labeled {IM_END} targets: {n_im_end}", flush=True)
    if tool_windows and im_end_id is not None:
        assert n_im_end > 0, (
            f"no labeled {IM_END} targets — the stop-token masking bug is back "
            "(spans must extend through m.end(0))")

    fp = corpus_fingerprint(ids_t, lbl_t)
    blob = {"ids": ids_t, "labels": lbl_t, "window": window,
            "im_end_targets": n_im_end, "fingerprint": fp, **extra}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out)
        print(f"[corpus] saved {ids_t.shape[0]} windows [{tuple(ids_t.shape)}] "
              f"fingerprint={fp} -> {out}", flush=True)
    print(f"[corpus] masked-token share overall: "
          f"{100 * (lbl_t != -100).float().mean():.1f}%", flush=True)
    return blob


# ------------------------------------------------------------------- log-based corpus
def build_log_corpus(*, window: int = 4096, wiki_tokens: int = 300_000,
                     data_split: str = "train", max_tool_tokens: int = 0,
                     min_density: float = 0.0, out: Path | None = None,
                     tok=None) -> dict:
    """The turn-aware assistant-masked corpus from a logtrain slice (+ optional wiki)."""
    tok = tok or load_tokenizer()
    sessions = [json.loads(line) for line in LOGTRAIN.read_text().splitlines()]
    splits = split.split_sessions(sessions, seed=42)
    picked = splits[data_split]
    print(f"[corpus] logtrain: {len(sessions)} sessions -> {data_split} slice {len(picked)} "
          f"(train={len(splits['train'])} test={len(splits['test'])} "
          f"holdout={len(splits['holdout'])})", flush=True)

    ids_stream: list[int] = []
    lbl_stream: list[int] = []
    tot_asst = n_real = n_stub = n_trunc = trunc_saved = 0
    for k, s in enumerate(picked):
        msgs = parse_session(s)
        msgs, n_tr, saved = truncate_tool_messages(msgs, tok, max_tool_tokens)
        n_trunc += n_tr
        trunc_saved += saved
        tools = split.session_tools(s, msgs)
        if tools:
            n_real += 1
        else:
            tools = reconstruct_tools(msgs)
            n_stub += 1
        ids, lbl = masked_ids_for_session(msgs, tok, tools=tools)
        ids_stream += ids
        lbl_stream += lbl
        tot_asst += sum(1 for x in lbl if x != -100)
        if (k + 1) % 25 == 0:
            print(f"[corpus]   {k+1}/{len(picked)} sessions, {len(ids_stream):,} tokens", flush=True)
    tool_windows = pack(ids_stream, lbl_stream, window, min_density=min_density)
    frac = 100 * tot_asst / max(1, len(ids_stream))
    print(f"[corpus] tool: {len(ids_stream):,} tokens ({frac:.0f}% assistant-masked) "
          f"-> {len(tool_windows)} windows of {window} (min_density={min_density})", flush=True)
    print(f"[corpus] schemas: {n_real} real, {n_stub} reconstructed stubs", flush=True)
    if max_tool_tokens > 0:
        print(f"[corpus] tool-output truncation: {n_trunc} messages, "
              f"{trunc_saved:,} masked tokens dropped", flush=True)

    wiki_windows: list[dict] = []
    if wiki_tokens > 0 and WIKI.exists():
        wids = tok(WIKI.read_text(), add_special_tokens=False)["input_ids"][:wiki_tokens]
        wiki_windows = pack(wids, list(wids), window)
        print(f"[corpus] wiki: {len(wids):,} tokens -> {len(wiki_windows)} full-loss windows", flush=True)

    windows = tool_windows + wiki_windows
    if not windows:
        sys.exit("[corpus] no windows survived packing — check --min-density / the slice")
    return _finalize(windows, window, tok, tool_windows=len(tool_windows), out=out,
                     extra={"tool_windows": len(tool_windows), "wiki_windows": len(wiki_windows),
                            "assistant_frac": frac / 100, "split": data_split,
                            "min_density": min_density, "max_tool_tokens": max_tool_tokens})


# ------------------------------------------------------------- distillation converter
def _text_of(item: dict) -> str:
    """Assistant 'message' content is a list of {text, ...} blocks (or a str)."""
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def trajectory_to_messages(items: list[dict],
                           system_prompt: str = SWE_SYSTEM_PROMPT) -> list[dict]:
    """OpenAI-Agents item list -> Qwen chat messages.

    Collapses an optional reasoning ``message`` + the following ``function_call``(s)
    into one assistant turn; each ``function_call_output`` becomes a ``role=tool``
    message. The leading ``role=user`` item is the problem statement.
    """
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    pending_text = ""
    pending_calls: list[dict] = []

    def flush_assistant():
        nonlocal pending_text, pending_calls
        if pending_text.strip() or pending_calls:
            m: dict = {"role": "assistant", "content": pending_text}
            if pending_calls:
                m["tool_calls"] = pending_calls
            msgs.append(m)
        pending_text, pending_calls = "", []

    for it in items:
        typ = it.get("type")
        if it.get("role") == "user" and typ != "function_call_output":
            flush_assistant()
            msgs.append({"role": "user", "content": _text_of(it) or it.get("content", "")})
        elif typ == "message":
            if pending_calls:
                flush_assistant()
            pending_text += _text_of(it)
        elif typ == "function_call":
            args = it.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            pending_calls.append({"type": "function",
                                  "function": {"name": it.get("name", "bash"),
                                               "arguments": args}})
        elif typ == "function_call_output":
            flush_assistant()
            out = it.get("output")
            msgs.append({"role": "tool",
                         "content": out if isinstance(out, str) else json.dumps(out)})
    flush_assistant()
    return msgs


def resolved_instance_ids(results_csv: Path) -> set[str]:
    ids: set[str] = set()
    with Path(results_csv).open() as f:
        for row in csv.DictReader(f):
            if str(row.get("resolved", "")).strip().lower() == "true":
                ids.add(row["instance_id"])
    return ids


def resolved_from_result_jsons(traj_dir: Path) -> set[str]:
    """Resolved instance_ids from the per-instance ``<inst>.result.json`` sidecars.

    The eval writes ``results.csv`` only at the very END of a run; the per-instance
    result.json is written as each instance is graded. This lets the distill corpus be
    built from a run that is still in flight or crashed before the CSV was emitted —
    same source of truth (``grade.resolved``), just per-file."""
    ids: set[str] = set()
    for rj in Path(traj_dir).glob("*.result.json"):
        try:
            if bool(json.loads(rj.read_text()).get("resolved")):
                ids.add(rj.name[: -len(".result.json")])
        except Exception:
            continue
    return ids


def build_distill_corpus(*, traj_dirs: list[Path], results: list[Path] | None = None,
                         all_patched: bool = False, window: int = 4096,
                         max_tool_tokens: int = 1024, min_density: float = 0.0,
                         out: Path | None = None, tok=None) -> dict:
    """Distill corpus from a strong solver's VERIFIED trajectories (resolved-only).

    ``results`` (a results.csv per traj_dir) is optional: when omitted (or an entry is
    None/missing), resolved status is read from the per-instance ``<inst>.result.json``
    sidecars in the traj_dir, so the corpus can be built from an in-flight/crashed run."""
    results = results or [None] * len(traj_dirs)
    if len(traj_dirs) != len(results):
        sys.exit("[distill] traj_dirs and results must be the same length")
    tok = tok or load_tokenizer()

    ids_stream: list[int] = []
    lbl_stream: list[int] = []
    tot_asst = n_kept = n_skipped = n_trunc = trunc_saved = 0
    per_source: list[str] = []

    for traj_dir, res in zip(traj_dirs, results, strict=True):
        traj_dir = Path(traj_dir)
        if all_patched:
            keep = None
        elif res is not None and Path(res).exists():
            keep = resolved_instance_ids(res)
            print(f"[distill] {Path(res).name}: {len(keep)} resolved instances", flush=True)
        else:
            keep = resolved_from_result_jsons(traj_dir)
            print(f"[distill] {traj_dir.name}: {len(keep)} resolved (from result.json sidecars)",
                  flush=True)
        traj_files = sorted(traj_dir.glob("*.traj.json"))
        kept_here = 0
        for tf in traj_files:
            inst = tf.name[: -len(".traj.json")]
            if keep is not None and inst not in keep:
                n_skipped += 1
                continue
            blob = json.loads(tf.read_text())
            msgs = trajectory_to_messages(blob["messages"])
            if not any(m["role"] == "assistant" for m in msgs):
                n_skipped += 1
                continue
            msgs, n_tr, saved = truncate_tool_messages(msgs, tok, max_tool_tokens)
            n_trunc += n_tr
            trunc_saved += saved
            ids, lbl = masked_ids_for_session(msgs, tok, tools=[BASH_TOOL])
            ids_stream += ids
            lbl_stream += lbl
            tot_asst += sum(1 for x in lbl if x != -100)
            n_kept += 1
            kept_here += 1
        per_source.append(f"{traj_dir.parent.parent.name}:{kept_here}")
        print(f"[distill]   {traj_dir}: kept {kept_here}/{len(traj_files)} trajectories", flush=True)

    if not ids_stream:
        sys.exit("[distill] no trajectories kept — check --results resolved flags / --traj-dir")
    windows = pack(ids_stream, lbl_stream, window, min_density=min_density)
    if not windows:
        sys.exit("[distill] no windows survived packing — lower --min-density")
    frac = 100 * tot_asst / max(1, len(ids_stream))
    print(f"[distill] {n_kept} trajectories ({n_skipped} skipped) -> {len(ids_stream):,} tokens "
          f"({frac:.0f}% assistant-masked) -> {len(windows)} windows of {window}", flush=True)
    if max_tool_tokens > 0:
        print(f"[distill] tool-output truncation: {n_trunc} messages, "
              f"{trunc_saved:,} masked tokens dropped", flush=True)
    return _finalize(windows, window, tok, tool_windows=len(windows), out=out,
                     extra={"tool_windows": len(windows), "wiki_windows": 0,
                            "assistant_frac": frac / 100, "split": "distill-swebench",
                            "min_density": min_density, "max_tool_tokens": max_tool_tokens,
                            "n_trajectories": n_kept, "sources": per_source,
                            "resolved_only": not all_patched})
