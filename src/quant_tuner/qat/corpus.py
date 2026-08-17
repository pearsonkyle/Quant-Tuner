"""QAT masked-corpus construction (log-based + strong-solver distillation).

Three corpora, one masking convention:

* :func:`build_log_corpus` — turn-aware, assistant-MASKED windows from the CLI-log
  slices (the iter-2/3/4 corpus). Renders each session with the model's real chat
  template, masks loss to the ``<|im_start|>assistant … <|im_end|>`` spans INCLUDING the
  terminating ``<|im_end|>`` (the stop decision — without it no position ever trains
  end-of-turn, the mechanistic cause of the iter-2/3 looping), and packs to windows.
* :func:`build_distill_corpus` — the iter-5 data-quality lever: re-render a strong
  solver's VERIFIED (tests-pass) SWE-rebench trajectories through the student's tokenizer
  with the same masking. Keeps resolved-only by default (a patch that doesn't pass is the
  mimicry trap iter-4 fell into). Data/trajectory distillation, not logit-KD (the solver's
  vocab differs); same-vocab logit-KD is a composable trainer lever (:mod:`.train`).
* :func:`build_sft_corpus` — the universal-corpus path: read the ``sft.jsonl.gz`` that
  ``quant_tuner.data.universal`` writes alongside the calibration corpus (FULL
  conversations, real tool schemas, system prompts already boilerplate-scrubbed, a
  ``split`` field that matches the calibration corpus) and mask/pack it the same way.
  Superset of the other two: the CLI logs, the agent logs, the verified SWE
  trajectories, the red-team refusals and the broad-instruct breadth in one file, with
  per-source token budgets so one source can't swamp the mix.

All three share the masking primitives (:func:`masked_ids_for_session`, :func:`pack`,
:func:`corpus_fingerprint`) so the two corpora are bit-for-bit trainer-compatible and the
stop-token/density audits are identical. The CLI shims are ``scripts/build_qat_masked_corpus.py``,
``scripts/build_ornith_distill_corpus.py`` and ``scripts/build_sft_qat_corpus.py``.
"""

from __future__ import annotations

import collections
import csv
import gzip
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
LOGTRAIN = REPO / "datasets" / "agent-logs" / "data" / "logs-cli.jsonl.gz"
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


#: Control tokens that must never appear inside message CONTENT. The corpus is rendered
#: with special-token parsing on (it has to be — that is how the chat structure works), so
#: a log that quotes one of these gets a REAL control token in the middle of prose.
CONTROL_TOKEN_STRINGS = ("<|im_start|>", "<|im_end|>")


def merge_consecutive_assistant(msgs: list[dict]) -> tuple[list[dict], int]:
    """Merge adjacent assistant messages into one turn. Returns (msgs, n_merged).

    THE DEFECT THIS FIXES. Agent logs record one logical assistant turn as several
    messages — a prose message ("Let me check the current state:") followed by a separate
    message carrying only ``tool_calls``. Rendered verbatim, each fragment becomes its own
    ``<|im_start|>assistant ... <|im_end|>`` turn, so the corpus teaches that a short
    prose preamble is followed by the STOP token rather than by the tool call that
    actually came next. Measured on the universal SFT corpus: 21.2% of logs-agents
    assistant messages and 9.0% of logs ones are preceded by another assistant message,
    and 18.5% of "Let me..."-opening assistant turns end at their first sentence, against
    0.0% in both the ultrachat and distillation corpora.

    That is the corpus half of the termination collapse (P(stop | sentence end) 0.95 in
    trained models against the shipped model's 0.009).

    Only ASSISTANT messages are merged. Tool results arrive as ``user``-role messages in
    this template, so merging consecutive user messages would fuse a real user turn with a
    tool response — a different and worse corruption.
    """
    out: list[dict] = []
    merged = 0
    for m in msgs:
        if (out and m.get("role") == "assistant" and out[-1].get("role") == "assistant"):
            prev = out[-1]
            a, b = (prev.get("content") or ""), (m.get("content") or "")
            # Two prose fragments are separate paragraphs of one turn; prose followed by a
            # tool-call-only message must not gain a trailing separator.
            prev["content"] = f"{a}\n\n{b}" if (a.strip() and b.strip()) else (a or b)
            calls = list(prev.get("tool_calls") or []) + list(m.get("tool_calls") or [])
            if calls:
                prev["tool_calls"] = calls
            ra, rb = (prev.get("reasoning_content") or ""), (m.get("reasoning_content") or "")
            if ra.strip() or rb.strip():
                prev["reasoning_content"] = f"{ra}\n\n{rb}" if (ra.strip() and rb.strip()) \
                    else (ra or rb)
            merged += 1
            continue
        out.append(dict(m))
    return out, merged


def drop_empty_assistant(msgs: list[dict]) -> tuple[list[dict], int]:
    """Remove assistant messages with no content and no tool calls.

    An empty assistant message renders as ``<|im_start|>assistant\\n<|im_end|>`` — a
    supervised span whose ONLY trained token is the stop token. It is the purest possible
    lesson in "emit <|im_end|> immediately", and the agent logs carry 2,155 of them
    (2,065 in logs-agents alone) from turns where the harness recorded a response that
    held only tool-use blocks, or nothing at all.

    That is what the `start` probe measures, and it moved 0.00002 -> 0.12194 in the run
    trained on this corpus.

    Run AFTER :func:`merge_consecutive_assistant`, which absorbs all but a handful of
    these into the neighbouring turn; this catches the ones with no assistant neighbour.
    """
    out = [m for m in msgs
           if not (m.get("role") == "assistant"
                   and not (m.get("content") or "").strip()
                   and not (m.get("tool_calls") or []))]
    return out, len(msgs) - len(out)


def has_inline_control_tokens(msgs: list[dict]) -> bool:
    """True if any message CONTENT quotes a chat control token.

    These come from our own past sessions debugging chat templates: the assistant wrote
    ``rendered.find('<|im_end|>')`` in a code block, and with special-token parsing on that
    becomes a real stop token inside supervised prose. Rare (2 of 6,645 occurrences) but
    unambiguously teaches the model to emit a control token mid-sentence, so the
    conversation is dropped rather than repaired — rewriting the token would corrupt the
    code the message is about.
    """
    for m in msgs:
        blob = (m.get("content") or "") + (m.get("reasoning_content") or "")
        if any(t in blob for t in CONTROL_TOKEN_STRINGS):
            return True
    return False


def masked_ids_for_session(
    msgs: list[dict], tok, tools: list | None = None, text: str | None = None
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
    if text is None:  # callers that already rendered pass it in to avoid a second render
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
            # the id, not just the count: `--stop-weight` needs to build a per-vocab CE
            # weight vector, and reading it from the blob keeps that independent of which
            # tokenizer the trainer happens to load
            "im_end_id": im_end_id, "im_end_targets": n_im_end, "fingerprint": fp, **extra}
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
    """The turn-aware assistant-masked corpus from a CLI-log slice (+ optional wiki)."""
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


# ------------------------------------------------------------------- universal SFT corpus
#: Per-source token budgets for :func:`build_sft_corpus`. **Empty by default — every
#: source is taken WHOLE.** This is the deliberate difference from the calibration
#: corpus: calibration is budgeted (~4.4M tokens, see ``data.universal``) because
#: llama-imatrix/AWQ/GPTQ sample a fixed slice and an unbalanced mix skews `E[a²]`;
#: QAT wants every sample it can get and spends its budget in *epochs*, not in tokens
#: on disk. Pass ``--budget SOURCE=N`` to cap one source (``0`` drops it).
SFT_DEFAULT_BUDGETS: dict[str, int | None] = {}


def read_sft_jsonl(path: Path) -> list[dict]:
    """Read ``sft.jsonl`` / ``sft.jsonl.gz`` (the ``data.universal`` SFT export)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        return [json.loads(line) for line in f if line.strip()]


_THINK_RE = re.compile(r"<think>\n(.*?)\n</think>", re.DOTALL)


def _sft_conversation_ids(conv: dict, tok, max_tool_tokens: int) -> tuple[list[int], list[int], dict]:
    """One SFT row -> (ids, labels, audit), rendered with the STUDENT's chat template.

    ``audit`` records what preprocessing COST, per conversation: tool-output tokens
    dropped by truncation, tool_calls that reached the render, and reasoning turns that
    survived it. Reasoning survival is not a given — the template keeps
    ``reasoning_content`` only on assistant turns AFTER the last user turn, which is the
    whole conversation for an agentic trajectory (one task turn, then dozens of
    assistant/tool turns) but only the tail segment of a multi-turn chat log.
    """
    msgs = [dict(m) for m in conv["messages"]]
    # BEFORE truncation and before rendering: merging changes which messages exist, so
    # doing it after would truncate and mask a message layout the template never sees.
    msgs, n_merged = merge_consecutive_assistant(msgs)
    # After merging: an assistant message with neither content nor tool calls renders
    # as a turn whose only supervised token is <|im_end|>. Merging absorbs all but a
    # handful; this removes the rest.
    msgs, n_empty = drop_empty_assistant(msgs)
    msgs, n_tr, saved = truncate_tool_messages(msgs, tok, max_tool_tokens)
    tools = conv.get("tools") or None
    if tools is None:
        # Only the tool-less sources (broad-instruct, refusals) land here; reconstructing
        # from observed calls would invent schemas for a conversation that has none.
        tools = reconstruct_tools(msgs) or None
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                   add_generation_prompt=False)
    ids, lbl = masked_ids_for_session(msgs, tok, tools=tools, text=text)
    audit = {
        "assistant_msgs_merged": n_merged,
        "empty_assistant_dropped": n_empty,
        "tool_msgs_truncated": n_tr,
        "tool_tokens_dropped": saved,
        "src_tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs),
        "rendered_tool_calls": text.count("<tool_call>"),
        # reasoning arrives either as a field (agent logs) or inline in content (CLI logs)
        "src_reasoning": sum(1 for m in msgs
                             if (m.get("reasoning_content") or "").strip()
                             or "</think>" in (m.get("content") or "")),
        "rendered_reasoning": sum(1 for m in _THINK_RE.finditer(text) if m.group(1).strip()),
    }
    return ids, lbl, audit


def build_sft_corpus(*, sft_path: Path, data_split: str | None = "train",
                     sources: list[str] | None = None,
                     budgets: dict[str, int | None] | None = None,
                     window: int = 4096, max_tool_tokens: int = 1024,
                     min_density: float = 0.0, seed: int = 42,
                     out: Path | None = None, tok=None) -> dict:
    """Masked QAT corpus from the universal ``sft.jsonl.gz``.

    Rows are filtered to ``data_split`` (``None`` = every split — don't, the eval
    holdouts live in the same file) and to ``sources``, shuffled per source with
    ``seed``, then consumed until that source's token budget is spent. Each source is
    packed into windows SEPARATELY so a window never glues two sources together; the
    combined window list is shuffled by :func:`_finalize`.
    """
    tok = tok or load_tokenizer()
    budgets = SFT_DEFAULT_BUDGETS if budgets is None else budgets
    rows = read_sft_jsonl(sft_path)
    if data_split is not None:
        rows = [r for r in rows if r.get("split") == data_split]
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_source[r.get("source", "?")].append(r)
    if sources:
        missing = [s for s in sources if s not in by_source]
        if missing:
            sys.exit(f"[sft] no {data_split!r} rows for source(s) {missing}; "
                     f"available: {sorted(by_source)}")
        by_source = {s: by_source[s] for s in sources}
    # budget 0 = drop the source entirely (the shim's way of excluding one)
    by_source = {s: rs for s, rs in by_source.items() if budgets.get(s, None) != 0}

    windows: list[dict] = []
    # Which source each window came from. Sources are packed separately (a window never
    # glues two together), so this is exact — and it is what lets the trainer report loss
    # per source. Without it a 5-source corpus yields one undifferentiated loss curve.
    window_source: list[int] = []
    source_names: list[str] = sorted(by_source)
    per_source: dict[str, dict] = {}
    for src in sorted(by_source):
        convs = list(by_source[src])
        random.Random(seed).shuffle(convs)
        budget = budgets.get(src, None)
        ids_stream: list[int] = []
        lbl_stream: list[int] = []
        n_conv = 0
        au: collections.Counter = collections.Counter()
        for conv in convs:
            # Dropped, not repaired: these are conversations that QUOTE a chat control
            # token in their content (our own past sessions debugging chat templates), so
            # the token becomes real inside supervised prose. Rewriting it would corrupt
            # the code the message is about, and there are only a handful.
            if has_inline_control_tokens(conv.get("messages") or []):
                au.update({"dropped_control": 1})
                continue
            ids, lbl, a = _sft_conversation_ids(conv, tok, max_tool_tokens)
            au.update(a)
            ids_stream += ids
            lbl_stream += lbl
            n_conv += 1
            if budget is not None and len(ids_stream) >= budget:
                break
        src_windows = pack(ids_stream, lbl_stream, window, min_density=min_density)
        n_all = len(pack(ids_stream, lbl_stream, window, min_density=0.0))
        asst = sum(1 for x in lbl_stream if x != -100)
        kept = len(ids_stream)
        per_source[src] = {
            "conversations_available": len(convs), "conversations_used": n_conv,
            "tokens": kept, "windows": len(src_windows),
            "windows_dropped_by_density": n_all - len(src_windows),
            "assistant_frac": round(asst / max(1, kept), 4),
            "budget": budget,
            # what preprocessing cost — never let these go silent
            "tool_tokens_dropped": au["tool_tokens_dropped"],
            "tool_truncation_share": round(
                au["tool_tokens_dropped"] / max(1, kept + au["tool_tokens_dropped"]), 4),
            "tool_msgs_truncated": au["tool_msgs_truncated"],
            "assistant_msgs_merged": au.get("assistant_msgs_merged", 0),
            "empty_assistant_dropped": au.get("empty_assistant_dropped", 0),
            "conversations_dropped_control_tokens": au.get("dropped_control", 0),
            "tool_calls_rendered": au["rendered_tool_calls"],
            "tool_calls_in_source": au["src_tool_calls"],
            "reasoning_rendered": au["rendered_reasoning"],
            "reasoning_in_source": au["src_reasoning"],
        }
        print(f"[sft] {src:<18} {n_conv}/{len(convs)} convs  {kept:>10,} tok  "
              f"{100 * asst / max(1, kept):4.0f}% masked  "
              f"-> {len(src_windows)} windows (-{n_all - len(src_windows)} low-density)",
              flush=True)
        print(f"[sft]   {'':<16} tool-calls {au['rendered_tool_calls']}/{au['src_tool_calls']} "
              f"kept · reasoning {au['rendered_reasoning']}/{au['src_reasoning']} kept · "
              f"tool-output truncation dropped {au['tool_tokens_dropped']:,} tok "
              f"({100 * per_source[src]['tool_truncation_share']:.0f}% of this source's "
              f"conversation content)", flush=True)
        if (au.get("assistant_msgs_merged") or au.get("dropped_control")
                or au.get("empty_assistant_dropped")):
            print(f"[sft]   {'':<16} merged {au.get('assistant_msgs_merged', 0):,} split "
                  f"assistant messages into their turn · dropped "
                  f"{au.get('empty_assistant_dropped', 0):,} empty assistant turn(s) "
                  f"(pure stop-token targets) · dropped "
                  f"{au.get('dropped_control', 0)} conversation(s) quoting a control token",
                  flush=True)
        windows += src_windows
        window_source += [source_names.index(src)] * len(src_windows)

    if not windows:
        sys.exit("[sft] no windows survived packing — check --split / --source / --min-density")
    tot_tok = sum(v["tokens"] for v in per_source.values())
    tot_asst = sum(v["tokens"] * v["assistant_frac"] for v in per_source.values())
    tot_drop = sum(v["tool_tokens_dropped"] for v in per_source.values())
    frac = tot_asst / max(1, tot_tok)
    print(f"[sft] TOTAL {tot_tok:,} tokens ({100 * frac:.0f}% assistant-masked) "
          f"-> {len(windows)} windows of {window}", flush=True)
    print(f"[sft] tool-calls {sum(v['tool_calls_rendered'] for v in per_source.values())}"
          f"/{sum(v['tool_calls_in_source'] for v in per_source.values())} · "
          f"reasoning {sum(v['reasoning_rendered'] for v in per_source.values())}"
          f"/{sum(v['reasoning_in_source'] for v in per_source.values())} · "
          f"tool-output truncation dropped {tot_drop:,} tok "
          f"({100 * tot_drop / max(1, tot_tok + tot_drop):.0f}% of conversation content) "
          f"at --max-tool-tokens {max_tool_tokens}", flush=True)
    return _finalize(windows, window, tok, tool_windows=len(windows), out=out,
                     extra={"tool_windows": len(windows), "wiki_windows": 0,
                            "assistant_frac": frac, "split": f"sft:{data_split}",
                            "min_density": min_density, "max_tool_tokens": max_tool_tokens,
                            "sft_path": str(sft_path), "per_source": per_source,
                            "source_names": source_names,
                            "window_source": torch.tensor(window_source, dtype=torch.int16)})
