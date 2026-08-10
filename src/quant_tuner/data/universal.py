"""The universal calibration corpus: everything in ``datasets/`` + wiki, one interleaved file.

``scripts/build_corpora.py`` calibrates on two sources (CLI logs + wiki). This builder is its
superset and the default for new models: every dataset the repo owns, so a quant is
calibrated on the same distribution mix we evaluate and train against.

| source                                       | role           | cal slice       | eval slice               |
|----------------------------------------------|----------------|-----------------|--------------------------|
| ``logs-cli.jsonl.gz`` (CLI usage logs)        | tool-calling   | train 80%       | holdout 10% → tools eval |
| ``logs-agents.jsonl.gz`` (agent trajectories) | 19 languages   | train 80%       | holdout 10% → tools eval |
| ``swe-agentic-trajectories``                  | long agentic   | 90% of resolved | 10% → agentic eval       |
| ``broad-domain-supplement``                   | breadth        | ``calib`` half  | 10% of ``mtp`` half      |
| ``redteam-safety-disclosures``                | **refusal**    | 90%, refused    | 10% → refusal eval       |
| ``wiki.test.raw``                             | general prose  | all             | —                        |
| ``eaddario/imatrix-calibration`` (external)   | —              | —               | code/math/tools + general|

Four properties this file exists to guarantee:

**1. Tool calls carry their structure.** The chat-shaped sources are rendered through the
model's own template with ``tools=`` attached, so the corpus contains the real
``<tool_call>``-style markers, argument JSON and tool results the model conditions on at
inference. The template is checked against a tool-calling fixture before anything is built
(:mod:`quant_tuner.data.template_check`), and the finished corpus is *re-scanned* for those
markers, per source — a corpus whose tool-call marker count is zero is a hard error.

**2. Reasoning is normalized, and what survives is counted.** Reasoning arrives inline in
``content`` from one log file and as a ``reasoning_content`` field from the other;
:mod:`quant_tuner.data.reasoning` makes them one shape. Chat templates keep reasoning only on
the *final* assistant turn of a render (measured, see that module), so the audit records
reasoning blocks available vs. blocks that actually landed rather than assuming.

**3. The red-team attacks are paired with refusals, never with what the target said.**
:mod:`quant_tuner.data.refusals` keeps the attack turns and substitutes a generic refusal for
every assistant turn; the original completions and ``target_reasoning`` never reach a corpus.
Refusal behavior is measurably what low-bit quantization erodes first, so the attack
distribution belongs in calibration — the harmful responses do not.

**4. Every source is spread through the whole file, and cal never shares a row with eval.**
Sources are interleaved proportionally (:func:`~quant_tuner.data.split.interleave_many`),
never concatenated: AWQ and GPTQ sample a fixed token budget across the corpus, and a source
written as one contiguous block either eats that budget or misses it. Each source is split by
a stable group key first, and disjointness is asserted at the end, not assumed.

The ``mtp`` half of the broad supplement is deliberately left out of calibration entirely
except for the eval slice: it is reserved for MTP draft-head training, and a head trained on
text the trunk was calibrated on would flatter its own acceptance rate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_tuner.data import (
    external,
    ingest,
    reasoning,
    refusals,
    split,
    system_prompt,
)

REPO = Path(__file__).resolve().parents[3]

DEFAULT_LOG_FILES = ingest.DEFAULT_LOG_FILES     # logs-cli.jsonl.gz + logs-agents.jsonl.gz
DEFAULT_WIKI = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"

BROAD_DATASET = "pearsonkyle/broad-domain-supplement"
BROAD_SPLIT = "corpus"
SWE_DATASET = "pearsonkyle/swe-agentic-trajectories"
SWE_SPLIT = "resolved"

SOURCE_LOGS = "logs"
SOURCE_SWE = "swe-trajectories"
SOURCE_BROAD = "broad-supplement"
SOURCE_REDTEAM = refusals.SOURCE_REDTEAM
SOURCE_REASONING = "reasoning"
SOURCE_WIKI = "wiki"

ALL_SOURCES = (SOURCE_LOGS, SOURCE_REASONING, SOURCE_SWE, SOURCE_BROAD, SOURCE_REDTEAM,
               SOURCE_WIKI)


# --------------------------------------------------------------------------------- config
@dataclass
class UniversalConfig:
    """Budgets and inputs for :func:`build`. Token budgets are per-source *targets*."""

    out_dir: Path
    model_dir: Path                       # HF dir carrying the tokenizer + chat template

    # Both on-disk log corpora by default: the CLI usage logs AND the harvested
    # multi-language agent trajectories. They are split and packed together — the packer's
    # (source, length_bucket) round-robin is what keeps one from swamping the other.
    log_files: tuple[Path, ...] = DEFAULT_LOG_FILES
    wiki: Path | None = DEFAULT_WIKI
    broad_jsonl: Path | None = None       # local override; else pulled from the Hub
    swe_jsonl: Path | None = None
    redteam_jsonl: Path | None = None     # local staging only (the splits are unpublished)

    # Calibration budgets (tokens). ``cal_wiki_tokens=None`` uses ALL of wiki (what the
    # published two-source releases did); set it when the chat budgets are small, or wiki
    # dominates the mix — check ``token_share`` in the audit.
    #
    # Measured availability on the Qwen3.6 tokenizer (message content, before windowing):
    # logs-cli 14.2M + logs-agents 15.1M, swe-resolved 1.03M, broad calib-half 0.34M,
    # redteam-refused 0.09M, wiki 0.30M. The log corpora are the only ones where the budget
    # (not the data) binds — the 500k inherited from the published two-source runs used 1.7%
    # of them. Raised so a release-quality imatrix sees a real cross-section; everything
    # else is set at or above its ceiling, so those sources are consumed whole. Cost is
    # llama-imatrix wall-clock, which is linear in corpus tokens.
    # ``None`` means "take all of it" (same semantics as cal_wiki_tokens): the smaller
    # sources are exhausted well before any sane cap, so pinning a number there would only
    # invite silent truncation the day one of them grows.
    cal_logs_tokens: int | None = 2_000_000
    cal_swe_tokens: int | None = 1_000_000
    cal_broad_tokens: int | None = None
    # The refusal source is small by construction (348 short exchanges) and is meant to be a
    # *present minority*, not a big share: enough that refusal directions survive the
    # codebook, not so much that the quant learns to decline ordinary requests.
    cal_redteam_tokens: int | None = None
    # Extra windows cut so a reasoning turn lands LAST — the only position a chat template
    # keeps reasoning in. Without them the corpus has ~no reasoning at all; see
    # `reasoning_windows`. They deliberately re-render spans the `logs` windows also cover.
    cal_reasoning_tokens: int | None = 1_000_000
    max_reasoning_windows_per_session: int = 8
    cal_wiki_tokens: int | None = None

    # Validation corpus (AWQ cv scoring): in-domain logs + out-of-domain breadth.
    val_logs_tokens: int = 10_000
    val_broad_tokens: int = 10_000

    # Eval corpora (each gets its OWN baseline.kld and is benched independently).
    eval_tokens_per_domain: int = 30_000  # external code/math/tools
    general_eval_tokens: int = 30_000     # external combined_en_tiny
    tools_eval_tokens: int = 30_000       # on-disk logs holdout, windowed
    agentic_eval_tokens: int = 30_000     # swe trajectories holdout, windowed
    broad_eval_tokens: int = 30_000       # broad supplement mtp-half slice
    redteam_eval_tokens: int = 10_000     # held-out attacks + refusals

    # The context every calibrator will read this corpus at. It is a PACKING parameter, not
    # just a runtime flag: a window longer than ctx straddles a chunk boundary, and a window
    # much shorter than ctx means the calibrator sees two unrelated conversations glued
    # together in one context. Measured at ctx 4096 / cap 3500, 51% of log windows and 46% of
    # SWE windows ended exactly at the cap — i.e. agentic tool-call chains were being cut
    # mid-chain, which is precisely the structure a coding model needs to see whole.
    # Whatever this is set to, `llama-imatrix -c`, `awq.calibrate(ctx=)` and
    # `gptq.calibrate(ctx=)` must be given the SAME value.
    ctx: int = 8192
    # Window budget. None = derive from ctx with headroom for the system prefix + schemas
    # that every render carries, so no window can straddle a context boundary.
    per_session_cap: int | None = None
    ctx_headroom: int = 692
    system_prose_budget: int = 256
    full_prose_quota: int = 1
    max_windows_per_session: int = 8
    # SWE trajectories are far longer than a CLI session (~34 tool calls each), so the
    # shared cap of 8 windows leaves most of each trajectory on the floor — it bound the
    # source to 0.65M of its ~0.93M available. Own knob, since it is a property of the
    # source's length distribution, not a global packing preference.
    max_windows_per_swe_session: int = 32
    tool_schema_quota: int | None = 1
    # Agentic tool outputs are whole test-suite dumps; a single one can be 10k+ tokens and
    # would consume an entire window as one un-windowable message.
    max_tool_output_tokens: int = 512

    seed: int = 42
    swe_eval_fraction: float = 0.10
    broad_eval_fraction: float = 0.10
    redteam_eval_fraction: float = 0.10
    # How assistant reasoning is normalized before templating; see data.reasoning.
    # "drop" for a non-thinking target model.
    reasoning_policy: str = "auto"
    # Also emit sft.jsonl.gz: the same sources as FULL conversations (untrimmed, untemplated,
    # tool_calls + reasoning_content as fields) for supervised fine-tuning. Token counts are
    # off by default — they cost a second tokenization pass over ~30M tokens and are specific
    # to whichever tokenizer this build used.
    write_sft: bool = True
    sft_token_counts: bool = False
    # Strip agent-harness boilerplate from SFT system prompts, keeping any block grounded in
    # a path this conversation actually touches (see data.system_prompt). Measured on these
    # logs: 6.4M system-prompt chars -> 0.4M. The CALIBRATION corpus is left alone — the
    # packer already stubs system prose to `system_prose_budget` after a quota, which is the
    # right mechanism there.
    sft_scrub_system: bool = True
    sft_boilerplate_min_sessions: int = system_prompt.DEFAULT_MIN_SESSIONS
    require_tool_calls: bool = True       # fail if the built corpus has no tool-call markers
    # Tokenize the finished corpus once more to report (and gate on) what AWQ's and
    # GPTQ's strided token budgets actually sample out of it.
    sampled_coverage_check: bool = True
    strict_template: bool = True          # fail on blocking chat-template checks

    sources: tuple[str, ...] = ALL_SOURCES

    def enabled(self, name: str) -> bool:
        return name in self.sources

    @property
    def window_cap(self) -> int:
        """Max tokens per emitted window — explicit, or derived from ``ctx``."""
        if self.per_session_cap is not None:
            return self.per_session_cap
        return max(512, self.ctx - self.ctx_headroom)


# -------------------------------------------------------------------------------- loading
def _hub_jsonl(dataset: str, split_name: str, override: Path | None) -> list[dict]:
    """Rows of one published split: an explicit override, the local staging copy, or the Hub.

    The staged copy under ``datasets/<name>/data/<split>.jsonl`` is byte-identical to what
    was pushed (``dataset.py push`` uploads it verbatim and records its sha256), so
    preferring it keeps the build reproducible offline.
    """
    name = dataset.split("/")[-1]
    candidates = [override] if override else []
    candidates.append(REPO / "datasets" / name / "data" / f"{split_name}.jsonl")
    for c in candidates:
        if c and Path(c).exists():
            path = Path(c)
            break
    else:
        from huggingface_hub import hf_hub_download

        print(f"  fetching {dataset}:{split_name} from the Hub ...", file=sys.stderr)
        path = Path(hf_hub_download(
            repo_id=dataset, filename=f"data/{split_name}.jsonl", repo_type="dataset",
        ))
    with path.open() as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    print(f"  {dataset}:{split_name} -> {len(rows)} rows ({path})", file=sys.stderr)
    return rows


# Sentinel for an uncapped budget: large enough that stratified_pack / pack_raw_samples
# exhaust their input first, which is exactly what "all of it" means to them.
_NO_CAP = 1 << 60


def _budget(v: int | None) -> int:
    return _NO_CAP if v is None else v


def _budget_label(v: int | None) -> object:
    return "all" if v is None else v


def _stable_fraction(key: str, seed: int) -> float:
    """Uniform-ish [0,1) hash of ``key``. Row-stable: adding rows never re-splits old ones."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16]
    return int(h, 16) / float(1 << 64)


def swe_sessions(rows: Iterable[dict]) -> list[dict]:
    """Published trajectory rows -> packer sessions.

    ``messages`` already begins with the SWE system prompt (``trajectory_to_messages``
    puts it there) and ``tools`` is the single ``bash`` schema the agent was actually
    given, so the windows we calibrate on match the eval-time prompt shape exactly.
    """
    out: list[dict] = []
    for r in rows:
        msgs = r.get("messages") or []
        if not msgs:
            continue
        out.append({
            "source": SOURCE_SWE,
            "messages": msgs,
            "tools": r.get("tools") or [],
            "id": r.get("instance_id", ""),
            "score": 1.0,
        })
    return out


def clip_tool_messages(messages: list[dict], tok, max_tokens: int) -> tuple[list[dict], int]:
    """Head+tail truncate ``role=="tool"`` contents to ``max_tokens``. Returns (msgs, n_clipped).

    The calibration-side analogue of :func:`quant_tuner.qat.corpus.truncate_tool_messages`
    (which additionally maintains loss masks). Both ends are kept because that is where a
    tool result carries signal — the command echo / first failure at the head, the summary
    line and exit status at the tail; the middle of a 10k-line dump is filler that would
    otherwise crowd real tool-call turns out of the token budget.
    """
    if max_tokens <= 0:
        return messages, 0
    head = max_tokens // 2
    tail = max_tokens - head
    clipped = 0
    out: list[dict] = []
    for m in messages:
        c = m.get("content")
        if m.get("role") != "tool" or not isinstance(c, str) or not c:
            out.append(m)
            continue
        ids = tok(c, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            out.append(m)
            continue
        text = (tok.decode(ids[:head], skip_special_tokens=True)
                + "\n... [truncated] ...\n"
                + tok.decode(ids[-tail:], skip_special_tokens=True))
        out.append(m | {"content": text})
        clipped += 1
    return out, clipped


def reasoning_windows(
    session: dict, tok, cap_tokens: int, *, max_windows: int = 4,
    system_content: str | None = None,
) -> list[tuple[str, int]]:
    """Windows that **end** on an assistant turn carrying reasoning.

    Why this exists: chat templates keep reasoning only on the final assistant turn of a
    render and scrub it from history (measured on Qwen3.6 — see :mod:`.reasoning`). The
    ordinary packer cuts windows wherever the token budget lands, so on a real build only 2
    of 4,291 available reasoning turns survived into the corpus: the model's dominant output
    mode was effectively absent from the calibration it was quantized against.

    Ending a window exactly at a reasoning turn puts that turn in the one position the
    template preserves — and it is on-distribution, because it is precisely the context the
    model has while it is generating that reasoning. The window grows backwards from the
    reasoning turn to fill ``cap_tokens``, so the preceding conversation comes along.
    """
    from quant_tuner.data.reasoning import reasoning_of

    messages = ingest.normalize_messages(session.get("messages") or [])
    ingest.coerce_tool_call_arguments(messages)
    tools = split.session_tools(session, messages)
    if messages and messages[0].get("role") == "system":
        sys_msg = dict(messages[0])
        if system_content is not None:
            sys_msg["content"] = system_content
        prefix, body = [sys_msg], messages[1:]
    else:
        prefix, body = [], messages

    targets = [i for i, m in enumerate(body)
               if m.get("role") == "assistant" and reasoning_of(m)]
    if not targets:
        return []

    def render(lo: int, hi: int) -> tuple[str, int]:
        text = tok.apply_chat_template(prefix + body[lo:hi], tools=tools, tokenize=False)
        return text, len(tok(text, add_special_tokens=False)["input_ids"])

    out: list[tuple[str, int]] = []
    used_upto = -1
    for end in targets:
        if len(out) >= max_windows:
            break
        if end <= used_upto:          # already covered by the previous window
            continue
        # Largest start <= end whose render fits the cap; binary search on the start index.
        lo, hi, best = 0, end, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if body[mid].get("role") not in ("user", "assistant"):
                lo = mid + 1
                continue
            try:
                _, n = render(mid, end + 1)
            except Exception:         # strict template refuses this start — move right
                lo = mid + 1
                continue
            if n <= cap_tokens:
                best = mid
                hi = mid - 1          # reach further back while it still fits
            else:
                lo = mid + 1
        if best is None:
            continue
        try:
            text, n = render(best, end + 1)
        except Exception:
            continue
        if n > 0:
            out.append((text.strip(), n))
            used_upto = end
    return out


def pack_raw_samples(
    texts: list[str], tok, target_tokens: int, window_tokens: int, seed: int,
) -> tuple[list[str], int]:
    """Group raw (non-chat) samples into ~``window_tokens`` chunks up to ``target_tokens``.

    Shuffled with ``seed`` so a topic-ordered source (the supplement is authored
    area/subject at a time) doesn't put all of one subject in one region of the corpus.
    """
    import random

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    total = 0
    for i in order:
        if total >= target_tokens:
            break
        t = texts[i].strip()
        if not t:
            continue
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        if cur and cur_tok + n > window_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(t)
        cur_tok += n
        total += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks, total


# --------------------------------------------------------------------------- jsonl writers
def write_jsonl_gz(path: Path, records: Iterable[dict]) -> tuple[int, int]:
    """Stream ``records`` to a gzipped JSONL. Returns ``(n_records, bytes_written)``."""
    import gzip

    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n, path.stat().st_size


def sft_record(session: dict, *, split_name: str, source: str, tok=None,
               boilerplate: set[str] | None = None,
               generic_tokens: set[str] | None = None) -> dict:
    """One conversation as an SFT training row — **complete, nothing trimmed**.

    This is deliberately *not* the calibration view. The calibration corpus is windowed to
    ≤3.5k tokens, has its system prose and tool schemas stubbed after a quota, clips tool
    outputs to 512 tokens, and is flattened to rendered text against one model's chat
    template. All of that is right for fitting an importance matrix and wrong for training:
    here every message survives in full, with ``tool_calls``, ``tool_call_id``/``name`` and
    ``reasoning_content`` kept as **separate fields** rather than inlined, so a trainer can
    template them however it likes (and mask reasoning independently of the answer).

    Model-agnostic by design — no chat template is applied. ``n_tokens`` is filled only when
    a tokenizer is passed, and is then specific to that tokenizer.
    """
    msgs = ingest.normalize_messages(session.get("messages") or [])
    ingest.coerce_tool_call_arguments(msgs)
    msgs = reasoning.apply_policy(msgs, "field")
    scrub_stats: dict | None = None
    if boilerplate:
        msgs, scrub_stats = system_prompt.scrub_messages(
            msgs, boilerplate=boilerplate, generic=generic_tokens)

    keep = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")
    clean = [{k: m[k] for k in keep if m.get(k) not in (None, "")} for m in msgs]
    clean = [m for m in clean if m.get("role")]

    n_chars = sum(len(str(m.get("content") or "")) + len(str(m.get("reasoning_content") or ""))
                  for m in clean)
    rec = {
        "id": session.get("id") or ingest.session_fingerprint(session)[:16],
        "source": source,
        "split": split_name,
        "messages": clean,
        "tools": split.session_tools(session, msgs) or [],
        "n_messages": len(clean),
        "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in clean),
        "n_tool_results": sum(1 for m in clean if m.get("role") == "tool"),
        "n_reasoning": sum(1 for m in clean if m.get("reasoning_content")),
        "n_chars": n_chars,
    }
    if session.get("meta"):
        rec["meta"] = session["meta"]
    if scrub_stats and scrub_stats["dropped"]:
        rec["system_scrub"] = scrub_stats
    if tok is not None:
        texts = [str(m.get("content") or "") for m in clean]
        texts += [str(m.get("reasoning_content") or "") for m in clean
                  if m.get("reasoning_content")]
        rec["n_tokens"] = sum(
            len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    return rec


# ------------------------------------------------------------------------------ marker scan
def tool_call_marker_counts(text: str, markers: Iterable[str]) -> dict[str, int]:
    return {m: text.count(m) for m in markers if m in text}


def scan_special_tokens(text: str, tok) -> dict[str, Any]:
    """Which control/special tokens the corpus carries, and whether they survive tokenization.

    Two stacks read this file and they do not agree by default:

    * ``llama-imatrix`` is run with ``--parse-special`` on (see ``models.llama_cpp.imatrix``),
      so ``<|im_start|>`` is one control token — matching inference.
    * the HF-side calibrators (AWQ/GPTQ) tokenize the same bytes with ``transformers``, which
      also encodes in-text specials as single ids **unless** ``split_special_tokens=True``.

    If either stopped holding, the corpus would silently become a different distribution for
    that calibrator — the markers would shatter into ordinary BPE pieces. So the check is on
    the built corpus, not on a fixture: every special token actually present must round-trip
    to exactly one id.
    """
    declared = list(getattr(tok, "all_special_tokens", None) or [])
    declared += list(getattr(tok, "additional_special_tokens", None) or [])
    from quant_tuner.data.template_check import (
        KNOWN_TOOL_CALL_MARKERS,
        KNOWN_TOOL_RESPONSE_MARKERS,
    )
    candidates = sorted({s for s in declared + list(KNOWN_TOOL_CALL_MARKERS)
                         + list(KNOWN_TOOL_RESPONSE_MARKERS)
                         if isinstance(s, str) and s.startswith("<") and s in text})
    present: dict[str, int] = {}
    multi_id: list[str] = []
    for s in candidates:
        present[s] = text.count(s)
        if len(tok(s, add_special_tokens=False)["input_ids"]) != 1:
            multi_id.append(s)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return {
        "present": present,
        "multi_id": multi_id,
        "all_single_id": not multi_id,
        "non_ascii_chars": non_ascii,
        "non_ascii_share": round(non_ascii / max(1, len(text)), 6),
        "has_carriage_returns": "\r" in text,
    }


def sampled_coverage(
    chunks: list[str], sources: list[str], tok, budgets: dict[str, int], ctx: int = 512,
) -> dict:
    """What each HF-side calibrator's token budget actually sees of this corpus.

    AWQ and GPTQ do not read the whole file: ``calibrate/_ingest.sample_chunks`` strides a
    fixed budget across it (AWQ 65536 tokens, GPTQ 32768). A corpus can be beautifully
    balanced overall and still hand GPTQ a 32k slice with no tool calls in it — that is the
    failure this reports.

    The stride is not re-implemented here; the production sampler is run over an *index*
    tensor, so the reported windows are exactly the ones a calibrator would receive, and each
    sampled token is attributed back to the source it came from.
    """
    import bisect

    import torch

    from quant_tuner.calibrate._ingest import sample_chunks

    per_chunk = tok(chunks, add_special_tokens=False)["input_ids"]
    starts: list[int] = []
    flat: list[int] = []
    for ids in per_chunk:
        starts.append(len(flat))
        flat.extend(ids)
    ids_t = torch.tensor(flat, dtype=torch.long)
    out: dict[str, Any] = {"corpus_tokens": int(ids_t.numel()), "ctx": ctx}

    for name, budget in budgets.items():
        picked = sample_chunks(torch.arange(ids_t.numel()), ctx, budget)
        idx = torch.cat(picked) if picked else torch.empty(0, dtype=torch.long)
        by_source: Counter = Counter()
        for i in idx.tolist():
            by_source[sources[bisect.bisect_right(starts, i) - 1]] += 1
        sampled_text = tok.decode(ids_t[idx].tolist(), skip_special_tokens=False) \
            if idx.numel() else ""
        scan = _scan_tool_calls(sampled_text)
        n = max(1, int(idx.numel()))
        out[name] = {
            "budget_tokens": budget,
            "chunks": len(picked),
            "sampled_tokens": int(idx.numel()),
            "corpus_share": round(idx.numel() / max(1, ids_t.numel()), 4),
            "source_share": {k: round(v / n, 4) for k, v in by_source.most_common()},
            "sources_seen": len(by_source),
            "tool_calls_in_sample": scan["tool_call_marker_total"],
            "tool_results_in_sample": scan["tool_response_marker_total"],
        }
    return out


def _scan_tool_calls(text: str) -> dict[str, Any]:
    from quant_tuner.data.template_check import (
        KNOWN_TOOL_CALL_MARKERS,
        KNOWN_TOOL_RESPONSE_MARKERS,
    )

    calls = tool_call_marker_counts(text, KNOWN_TOOL_CALL_MARKERS)
    resps = tool_call_marker_counts(text, KNOWN_TOOL_RESPONSE_MARKERS)
    return {
        "tool_call_markers": calls,
        "tool_call_marker_total": sum(calls.values()),
        "tool_response_markers": resps,
        "tool_response_marker_total": sum(resps.values()),
    }


# ---------------------------------------------------------------------------------- build
@dataclass
class _Part:
    """One source's contribution to the calibration corpus."""

    name: str
    chunks: list[str]
    tokens: int
    audit: dict = field(default_factory=dict)


def _write_sft(
    cfg: UniversalConfig, out: Path, tok, *,
    log_splits: dict[str, list[dict]],
    swe_cal: list[dict], swe_eval: list[dict],
    redteam_cal: list[dict], redteam_eval: list[dict],
) -> dict:
    """Write ``sft.jsonl.gz``: every chat-shaped source, full fidelity, split-tagged.

    Re-reads the sources rather than reusing the packer's copies, because by this point
    those have been clipped and windowed for calibration. The ``split`` field carries the
    SAME assignment the calibration corpus used, so training on ``split == "train"`` leaves
    the tools / agentic / refusal eval holdouts genuinely held out.
    """
    counter = tok if cfg.sft_token_counts else None
    by_source: dict[str, dict[str, int]] = {}
    scrub_totals = {"conversations_scrubbed": 0, "blocks_dropped": 0,
                    "system_chars_before": 0, "system_chars_after": 0}

    # Boilerplate has to be known corpus-wide before the first record is written, so the log
    # sessions are loaded once up front. Everything else streams.
    log_sessions: list[dict] = []
    boilerplate: set[str] = set()
    generic: set[str] = set()
    if cfg.enabled(SOURCE_LOGS):
        log_sessions = ingest.filter_sessions(
            ingest.load_all_sessions([ingest.resolve_log_path(p) for p in cfg.log_files]),
            min_score=0.3, require_tools=False,
        )
    if cfg.sft_scrub_system and log_sessions:
        norm = [ingest.normalize_messages(s.get("messages") or []) for s in log_sessions]
        boilerplate = system_prompt.boilerplate_blocks(
            [c for m in norm if (c := system_prompt.system_content_of(m))],
            min_sessions=cfg.sft_boilerplate_min_sessions,
        )
        generic = system_prompt.generic_path_tokens(system_prompt.body_text(m) for m in norm)
        print(f"  sft: {len(boilerplate)} boilerplate system blocks, "
              f"{len(generic)} generic path tokens", file=sys.stderr)

    def _rec(session: dict, *, split_name: str, source: str) -> dict:
        rec = sft_record(session, split_name=split_name, source=source, tok=counter,
                         boilerplate=boilerplate, generic_tokens=generic)
        if (st := rec.get("system_scrub")):
            scrub_totals["conversations_scrubbed"] += 1
            scrub_totals["blocks_dropped"] += st["dropped"]
            scrub_totals["system_chars_before"] += st["chars_before"]
            scrub_totals["system_chars_after"] += st["chars_after"]
        return _tally(by_source, rec)

    def records() -> Iterable[dict]:
        # logs: re-load (the packer's copies are clipped by now), then tag from the split
        # calibration used. Matched on the row `id`, which both log formats carry and which
        # is unaffected by normalization — session_fingerprint is NOT usable here, because
        # it hashes messages[0] and the packer's copies have theirs parsed from JSON
        # strings into dicts, so every CLI session would silently fail to match.
        split_of: dict[str, str] = {
            str(s.get("id") or ingest.session_group(s)): name
            for name, sessions in log_splits.items() for s in sessions
        }
        for s in log_sessions:
            name = split_of.get(str(s.get("id") or ingest.session_group(s)))
            if name is None:
                continue
            src = "logs-agents" if str(s.get("source", "")).startswith("agents:") \
                else SOURCE_LOGS
            yield _rec(s, split_name=name, source=src)

        if swe_cal or swe_eval:
            for rows, name in ((swe_cal, "train"), (swe_eval, "holdout")):
                ids = {s["id"] for s in rows}
                for r in _hub_jsonl(SWE_DATASET, SWE_SPLIT, cfg.swe_jsonl):
                    if r.get("instance_id") not in ids:
                        continue
                    s = {"id": r["instance_id"], "messages": r.get("messages") or [],
                         "tools": r.get("tools") or [],
                         "meta": {k: r.get(k) for k in ("repo", "resolved", "instance_id")}}
                    yield _rec(s, split_name=name, source=SOURCE_SWE)

        # red-team: the REFUSAL-substituted conversations, never the originals.
        for rows, name in ((redteam_cal, "train"), (redteam_eval, "holdout")):
            for s in rows:
                yield _rec(s, split_name=name, source=SOURCE_REDTEAM)

        # the supplement's instruction view — already prompt/response pairs
        for r in _hub_jsonl(BROAD_DATASET, "instruct", None):
            s = {"id": r.get("id"), "messages": r.get("messages") or [],
                 "meta": {k: r.get(k) for k in ("area", "subject", "register", "half",
                                                "prompt_source")}}
            # `half` mirrors the calibration split: calib was seen by the imatrix, mtp
            # was not (it is reserved for draft-head training).
            yield _rec(s, split_name="train" if r.get("half") == "calib" else "holdout",
                       source="broad-instruct")

    path = out / "sft.jsonl.gz"
    n, nbytes = write_jsonl_gz(path, records())
    print(f"  sft:         {path} ({n:,} conversations, {nbytes / 1024**2:.1f} MiB gz)",
          file=sys.stderr)
    return {
        "path": str(path),
        "conversations": n,
        "bytes_gz": nbytes,
        "per_source": by_source,
        "token_counts": cfg.sft_token_counts,
        "system_scrub": {
            "enabled": cfg.sft_scrub_system,
            "min_sessions": cfg.sft_boilerplate_min_sessions,
            "boilerplate_blocks": len(boilerplate),
            "generic_path_tokens": len(generic),
            **scrub_totals,
            "note": "harness boilerplate dropped from system prompts; a repeated block is "
                    "KEPT when it names a path/file this conversation actually touches "
                    "(data.system_prompt). The calibration corpus is not scrubbed — the "
                    "packer stubs system prose there instead.",
        },
        "fields": ["id", "source", "split", "messages", "tools", "meta", "n_messages",
                   "n_tool_calls", "n_tool_results", "n_reasoning", "n_chars"]
                  + (["n_tokens"] if cfg.sft_token_counts else []),
        "note": "FULL conversations — no windowing, no tool-output clipping, no system/"
                "schema stubbing, no chat template applied. tool_calls and "
                "reasoning_content are separate message fields. `split` matches the "
                "calibration corpus, so training on split=='train' keeps the eval "
                "holdouts held out.",
    }


def _tally(acc: dict[str, dict[str, int]], rec: dict) -> dict:
    a = acc.setdefault(rec["source"], {"conversations": 0, "messages": 0, "tool_calls": 0,
                                       "reasoning_turns": 0, "chars": 0, "tokens": 0})
    a["conversations"] += 1
    a["messages"] += rec["n_messages"]
    a["tool_calls"] += rec["n_tool_calls"]
    a["reasoning_turns"] += rec["n_reasoning"]
    a["chars"] += rec["n_chars"]
    a["tokens"] += rec.get("n_tokens", 0)
    return rec


def build(cfg: UniversalConfig) -> dict:
    """Build every corpus and return the audit dict (also written as corpora_audit.json)."""
    from transformers import AutoTokenizer

    from quant_tuner.data.template_check import check_template

    out = Path(cfg.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_dir, fix_mistral_regex=True)

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    # --- 0) the chat template must be able to carry tool calls at all ------------------
    report = check_template(tok, str(cfg.model_dir))
    print(report.summary(), file=sys.stderr)
    if cfg.strict_template and not report.ok:
        raise RuntimeError(
            "refusing to build: the chat template cannot render tool calls (see report "
            "above). Re-run with strict_template=False only if you have verified the "
            "corpus by hand."
        )

    audit: dict[str, Any] = {
        "builder": "quant_tuner.data.universal",
        "seed": cfg.seed,
        "model_dir": str(cfg.model_dir),
        "sources": list(cfg.sources),
        "template_check": report.to_dict(),
        "ctx": cfg.ctx,
        "window_cap": cfg.window_cap,
        "ctx_note": "every calibrator must read this corpus at ctx=%d — llama-imatrix -c, "
                    "awq.calibrate(ctx=), gptq.calibrate(ctx=). Windows are packed to fit "
                    "one context exactly." % cfg.ctx,
        "budgets": {
            "cal_logs_tokens": _budget_label(cfg.cal_logs_tokens),
            "cal_swe_tokens": cfg.cal_swe_tokens,
            "cal_broad_tokens": cfg.cal_broad_tokens,
        },
    }
    pack_kwargs: dict[str, Any] = dict(
        per_session_cap=cfg.window_cap,
        seed=cfg.seed,
        system_prose_budget=cfg.system_prose_budget,
        full_prose_quota=cfg.full_prose_quota,
        max_windows_per_session=cfg.max_windows_per_session,
        tool_schema_quota=cfg.tool_schema_quota,
    )

    parts: list[_Part] = []
    fp = ingest.session_fingerprint
    log_splits: dict[str, list[dict]] = {"train": [], "test": [], "holdout": []}
    swe_cal: list[dict] = []
    swe_eval: list[dict] = []
    broad_cal: list[str] = []
    broad_val: list[str] = []
    broad_eval: list[str] = []
    redteam_cal: list[dict] = []
    redteam_eval: list[dict] = []

    prep_stats: dict[str, dict] = {}

    def prepare(sessions: list[dict], label: str) -> list[dict]:
        """Normalize messages, reasoning and oversized tool outputs before packing."""
        clipped = 0
        available = 0
        for s in sessions:
            msgs = ingest.normalize_messages(s.get("messages") or [])
            msgs = reasoning.apply_policy(msgs, cfg.reasoning_policy)
            available += reasoning.count_available(msgs)
            msgs, n = clip_tool_messages(msgs, tok, cfg.max_tool_output_tokens)
            clipped += n
            s["messages"] = msgs
        prep_stats[label] = {
            "reasoning_policy": cfg.reasoning_policy,
            "reasoning_blocks_available": available,
            "tool_outputs_clipped": clipped,
            "max_tool_output_tokens": cfg.max_tool_output_tokens,
        }
        return sessions

    # --- 1) on-disk logs: CLI sessions + harvested agent trajectories -----------------
    if cfg.enabled(SOURCE_LOGS):
        present = [ingest.resolve_log_path(p) for p in cfg.log_files]
        missing = [str(p) for p in present if not p.exists()]
        assert not missing, f"missing log file(s): {missing}"
        sessions = prepare(ingest.filter_sessions(
            ingest.load_all_sessions(present), min_score=0.3, require_tools=False,
        ), SOURCE_LOGS)
        # Group-aware (see ingest.session_group): the agent logs hold several attempts at
        # the same issue, which must not straddle the cal/eval boundary.
        log_splits = split.split_sessions(
            sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=cfg.seed,
        )
        chunks, _kept, total, pack = split.stratified_pack(
            log_splits["train"], tok, target_tokens=_budget(cfg.cal_logs_tokens), **pack_kwargs,
        )
        split.write_corpus(chunks, out / "corpus.cal.logs.txt")
        by_file = {str(p): sum(1 for s in ingest.load_sessions(p)) for p in present}
        parts.append(_Part(SOURCE_LOGS, chunks, total, {
            "files": by_file,
            "sessions_after_filter": len(sessions),
            "sessions": {k: len(v) for k, v in log_splits.items()},
            "groups": {k: len({ingest.session_group(s) for s in v})
                       for k, v in log_splits.items()},
            "target_tokens": _budget_label(cfg.cal_logs_tokens),
            **prep_stats[SOURCE_LOGS],
            "pack_audit": pack,
        }))

    # --- 1b) reasoning-terminal windows ----------------------------------------------
    # Without this the corpus carries essentially no reasoning (see reasoning_windows).
    # None = uncapped, 0 = disabled — so test against 0 explicitly, not truthiness.
    if (cfg.enabled(SOURCE_REASONING) and log_splits["train"]
            and cfg.cal_reasoning_tokens != 0):
        import random as _random

        rchunks: list[str] = []
        rtotal = 0
        scanned = 0
        contributing = 0
        # Shuffle before spending the budget: in split order the first sessions that happen
        # to carry reasoning would absorb it entirely (measured: ~16 sessions for a 150k
        # budget), so the reasoning slice would describe a handful of conversations rather
        # than the corpus. Seeded, so rebuilds are identical.
        candidates = list(log_splits["train"])
        _random.Random(cfg.seed).shuffle(candidates)
        for s in candidates:
            if rtotal >= _budget(cfg.cal_reasoning_tokens):
                break
            scanned += 1
            wins = reasoning_windows(
                s, tok, cfg.window_cap,
                max_windows=cfg.max_reasoning_windows_per_session,
            )
            if wins:
                contributing += 1
            for text, n in wins:
                if rtotal >= _budget(cfg.cal_reasoning_tokens):
                    break
                rchunks.append(text)
                rtotal += n
        if rchunks:
            split.write_corpus(rchunks, out / "corpus.cal.reasoning.txt")
            parts.append(_Part(SOURCE_REASONING, rchunks, rtotal, {
                "target_tokens": _budget_label(cfg.cal_reasoning_tokens),
                "source": "logs train slice, windows ending on a reasoning turn",
                "max_windows_per_session": cfg.max_reasoning_windows_per_session,
                "sessions_scanned": scanned,
                "sessions_contributing": contributing,
                "note": "these overlap the `logs` windows by design — they re-render the "
                        "same conversations cut so the reasoning turn is LAST, which is the "
                        "only position a chat template preserves it in",
            }))

    # --- 2) published agentic trajectories -------------------------------------------
    if cfg.enabled(SOURCE_SWE):
        rows = _hub_jsonl(SWE_DATASET, SWE_SPLIT, cfg.swe_jsonl)
        sess = prepare(swe_sessions(rows), SOURCE_SWE)
        for s in sess:
            key = s.get("id") or fp(s)
            (swe_eval if _stable_fraction(key, cfg.seed) < cfg.swe_eval_fraction
             else swe_cal).append(s)
        chunks, _kept, total, pack = split.stratified_pack(
            swe_cal, tok, target_tokens=_budget(cfg.cal_swe_tokens),
            **{**pack_kwargs, "max_windows_per_session": cfg.max_windows_per_swe_session},
        )
        split.write_corpus(chunks, out / "corpus.cal.swe.txt")
        parts.append(_Part(SOURCE_SWE, chunks, total, {
            "dataset": f"{SWE_DATASET}:{SWE_SPLIT}",
            "rows": len(sess), "cal_rows": len(swe_cal), "eval_rows": len(swe_eval),
            "target_tokens": _budget_label(cfg.cal_swe_tokens),
            **prep_stats[SOURCE_SWE],
            "pack_audit": pack,
        }))

    # --- 3) broad-domain supplement ---------------------------------------------------
    if cfg.enabled(SOURCE_BROAD):
        rows = _hub_jsonl(BROAD_DATASET, BROAD_SPLIT, cfg.broad_jsonl)
        mtp_held = 0
        for r in rows:
            text = r.get("text") or ""
            if not text.strip():
                continue
            key = r.get("id") or hashlib.sha256(text.encode()).hexdigest()[:16]
            if r.get("half") == "calib":
                broad_cal.append(text)
            elif _stable_fraction(f"broad:{key}", cfg.seed) < cfg.broad_eval_fraction:
                broad_eval.append(text)
            else:
                mtp_held += 1
        # The validation slice is drawn from the EVAL side of the mtp half, then removed
        # from it: val is scored during the AWQ α search, so a row used there is no longer
        # a clean holdout for PPL/KLD.
        n_val = max(0, min(len(broad_eval) // 4, len(broad_eval) - 1))
        broad_val, broad_eval = broad_eval[:n_val], broad_eval[n_val:]
        chunks, total = pack_raw_samples(
            broad_cal, tok, _budget(cfg.cal_broad_tokens), cfg.window_cap,
            cfg.seed,
        )
        split.write_corpus(chunks, out / "corpus.cal.broad.txt")
        parts.append(_Part(SOURCE_BROAD, chunks, total, {
            "dataset": f"{BROAD_DATASET}:{BROAD_SPLIT}",
            "calib_half_rows": len(broad_cal),
            "eval_rows": len(broad_eval), "val_rows": len(broad_val),
            "mtp_half_rows_reserved": mtp_held,
            "target_tokens": _budget_label(cfg.cal_broad_tokens),
            "note": "the mtp half is reserved for MTP draft-head training and is NOT in "
                    "calibration; only its eval slice is used here",
        }))

    # --- 3b) red-team attacks, every response replaced by a generic refusal -----------
    if cfg.enabled(SOURCE_REDTEAM):
        rt_rows = refusals.load_rows(cfg.redteam_jsonl)
        if not rt_rows:
            print(f"  redteam: no staged rows at {refusals.REDTEAM_DIR} — source skipped",
                  file=sys.stderr)
        else:
            sess = list(refusals.refusal_sessions(rt_rows))
            for s in sess:
                (redteam_eval if _stable_fraction(f"rt:{s['id']}", cfg.seed)
                 < cfg.redteam_eval_fraction else redteam_cal).append(s)
            chunks, _kept, total, pack = split.stratified_pack(
                redteam_cal, tok, target_tokens=_budget(cfg.cal_redteam_tokens), **pack_kwargs,
            )
            split.write_corpus(chunks, out / "corpus.cal.redteam.txt")
            outcomes: dict[str, int] = {}
            for r in rt_rows:
                k = str(r.get("outcome"))
                outcomes[k] = outcomes.get(k, 0) + 1
            parts.append(_Part(SOURCE_REDTEAM, chunks, total, {
                "dataset": f"{refusals.REDTEAM_DATASET}:{refusals.REDTEAM_SPLIT}",
                "rows": len(rt_rows), "sessions": len(sess),
                "cal_rows": len(redteam_cal), "eval_rows": len(redteam_eval),
                "original_outcomes": outcomes,
                "target_ms_replaced": True,
                "target_reasoning_included": False,
                "n_refusal_templates": len(refusals.GENERIC_REFUSALS),
                "target_tokens": _budget_label(cfg.cal_redteam_tokens),
                "note": "attack turns kept verbatim; EVERY assistant turn replaced with a "
                        "generic refusal. The targets' original completions and "
                        "target_reasoning are never written to any corpus.",
                "pack_audit": pack,
            }))

    # --- 4) wiki ----------------------------------------------------------------------
    if cfg.enabled(SOURCE_WIKI) and cfg.wiki:
        assert Path(cfg.wiki).exists(), f"missing wiki: {cfg.wiki}"
        all_chunks = split.chunk_text(
            Path(cfg.wiki).read_text(), approx_chars=cfg.window_cap * 4,
        )
        wiki_chunks: list[str] = []
        wiki_total = 0
        for c in all_chunks:
            if cfg.cal_wiki_tokens is not None and wiki_total >= cfg.cal_wiki_tokens:
                break
            wiki_chunks.append(c)
            wiki_total += ntok(c)
        parts.append(_Part(SOURCE_WIKI, wiki_chunks, wiki_total, {
            "path": str(cfg.wiki), "n_chunks": len(wiki_chunks),
            "n_chunks_available": len(all_chunks),
            "target_tokens": cfg.cal_wiki_tokens,
        }))

    # --- 5) the calibration corpus: all sources interleaved ---------------------------
    cal_corpus = out / "corpus.cal.txt"
    ordered = split.interleave_many([p.chunks for p in parts])
    split.write_corpus(ordered, cal_corpus)
    # Same content, one record per window, with the source label attached. corpus.cal.txt
    # is what llama-imatrix eats; this is what you query when a number looks wrong and you
    # need to know which source a given span came from.
    source_of = {id(c): p.name for p in parts for c in p.chunks}
    write_jsonl_gz(
        out / "corpus.cal.jsonl.gz",
        ({"i": i, "source": source_of.get(id(c), "?"), "n_chars": len(c), "text": c}
         for i, c in enumerate(ordered)),
    )
    # newline="" so the scan sees the bytes on disk: universal-newline translation
    # rewrites the \r that agent tool output carries, and the audit would deny it exists.
    with open(cal_corpus, newline="") as _fh:
        cal_text = _fh.read()
    scan = _scan_tool_calls(cal_text)
    # Per-source too: a source that quietly stops carrying tool calls (a template change,
    # a schema-dedup regression) is invisible in the total, which the other chat source
    # keeps non-zero.
    for p in parts:
        joined = "\n\n".join(p.chunks)
        p.audit["tool_calls"] = _scan_tool_calls(joined)
        # Reasoning that SURVIVED templating, against what the source had. Chat templates
        # keep reasoning only on a render's final assistant turn, so this ratio is expected
        # to be well under 1 — but a 0 means the policy or the template silently ate it all.
        p.audit["reasoning_blocks_in_corpus"] = reasoning.count_reasoning_blocks(joined)
    audit["calibration"] = {
        "path": str(cal_corpus),
        "bytes": cal_corpus.stat().st_size,
        "total_tokens": sum(p.tokens for p in parts),
        "per_source": {p.name: {"tokens": p.tokens, "chunks": len(p.chunks), **p.audit}
                       for p in parts},
        "token_share": {p.name: round(p.tokens / max(1, sum(q.tokens for q in parts)), 4)
                        for p in parts},
        "tool_calls": scan,
        "reasoning": {
            "policy": cfg.reasoning_policy,
            "blocks_in_corpus": reasoning.count_reasoning_blocks(cal_text),
            "note": "chat templates keep reasoning only on a render's FINAL assistant turn, "
                    "so this is far below the number of reasoning turns in the sources "
                    "(see per_source.*.reasoning_blocks_available) — by design, since that "
                    "is also what the model sees at inference",
        },
    }
    for p in parts:
        print(f"  cal[{p.name}]: {p.tokens:,} tokens in {len(p.chunks)} chunks, "
              f"{p.audit['tool_calls']['tool_call_marker_total']} tool calls, "
              f"{p.audit['reasoning_blocks_in_corpus']} reasoning blocks",
              file=sys.stderr)
        # The log source spans ~20 agent sources (2 CLIs + 19 trajectory languages) and the
        # packer round-robins across them; print the spread so a source quietly dominating
        # (or vanishing) is visible without opening the audit.
        by_src = p.audit.get("pack_audit", {}).get("tokens_per_source") or {}
        if len(by_src) > 2:
            top = sorted(by_src.items(), key=lambda kv: -kv[1])
            share = ", ".join(f"{k} {v / max(1, p.tokens):.0%}" for k, v in top[:4])
            print(f"      across {len(by_src)} agent sources — top: {share}",
                  file=sys.stderr)
    print(f"  calibration: {cal_corpus} ({scan['tool_call_marker_total']} tool-call "
          f"markers, {scan['tool_response_marker_total']} tool results)", file=sys.stderr)

    # Special characters / control tokens, checked on the corpus rather than a fixture, and
    # what each HF-side calibrator's strided budget actually samples out of it.
    specials = scan_special_tokens(cal_text, tok)
    audit["calibration"]["special_tokens"] = specials
    if not specials["all_single_id"]:
        raise RuntimeError(
            "these control tokens present in the calibration corpus do NOT tokenize to a "
            f"single id: {specials['multi_id']}. AWQ/GPTQ would calibrate on shattered BPE "
            "pieces where inference sees one control token."
        )
    if cfg.sampled_coverage_check:
        audit["calibration"]["sampled_coverage"] = sampled_coverage(
            ordered, [source_of.get(id(c), "?") for c in ordered], tok,
            {"awq": 65_536, "gptq": 32_768})
        for name in ("awq", "gptq"):
            cov = audit["calibration"]["sampled_coverage"][name]
            print(f"  {name}: samples {cov['sampled_tokens']:,} tokens "
                  f"({cov['corpus_share']:.1%} of corpus) from {cov['sources_seen']} sources "
                  f"{cov['source_share']}, {cov['tool_calls_in_sample']} tool calls",
                  file=sys.stderr)
            if cfg.require_tool_calls and cov["tool_calls_in_sample"] == 0:
                raise RuntimeError(
                    f"{name}'s {cov['budget_tokens']}-token budget samples ZERO tool calls "
                    "from this corpus — it would calibrate on the wrong distribution. "
                    "Interleaving is what prevents this; check token_share in the audit."
                )

    if cfg.require_tool_calls and scan["tool_call_marker_total"] == 0:
        raise RuntimeError(
            "the calibration corpus contains ZERO recognised tool-call markers. Either the "
            "chat-shaped sources were disabled, or this template renders calls in a form "
            "we don't recognise — inspect corpus.cal.txt and extend "
            "template_check.KNOWN_TOOL_CALL_MARKERS before calibrating."
        )

    # --- 6) validation corpus (AWQ cv scoring) ----------------------------------------
    val_chunks: list[list[str]] = []
    val_audit: dict[str, Any] = {}
    if cfg.enabled(SOURCE_LOGS) and cfg.val_logs_tokens > 0:
        vc, _k, vtot, vpack = split.stratified_pack(
            log_splits["test"], tok, target_tokens=cfg.val_logs_tokens, **pack_kwargs,
        )
        val_chunks.append(vc)
        val_audit["logs_test"] = {"tokens": vtot, "chunks": len(vc), "pack_audit": vpack}
    if broad_val and cfg.val_broad_tokens > 0:
        bc, btot = pack_raw_samples(
            broad_val, tok, cfg.val_broad_tokens, cfg.window_cap, cfg.seed + 1,
        )
        val_chunks.append(bc)
        val_audit["broad_supplement"] = {"tokens": btot, "chunks": len(bc)}
    if val_chunks:
        val_corpus = out / "corpus.val.txt"
        split.write_corpus(split.interleave_many(val_chunks), val_corpus)
        audit["validation"] = {"path": str(val_corpus), "sources": val_audit,
                               "note": "in-domain logs + out-of-domain breadth, so an AWQ α "
                                       "that wins here generalizes past the log mix"}
        print(f"  validation:  {val_corpus}", file=sys.stderr)

    # --- 7) eval corpora — each SEPARATE, each with its own baseline.kld ---------------
    audit["eval"] = {}

    ext_corpus = out / "corpus.eval.txt"
    ext_corpus.write_text("")
    domains: dict[str, Any] = {}
    for i, domain in enumerate(external.EVAL_DOMAINS):
        text, n, rows_used = external.sample_parquet_text(
            external.download_parquet(domain), tok,
            target_tokens=cfg.eval_tokens_per_domain, seed=cfg.seed + i,
        )
        (out / f"corpus.eval.{domain}.txt").write_text(text)
        with ext_corpus.open("a") as fh:
            fh.write(text.rstrip() + "\n\n")
        domains[domain] = {"tokens": n, "rows": rows_used}
    audit["eval"]["external"] = {
        "path": str(ext_corpus), "repo": external.EAD_REPO, "domains": domains,
        "used_for": "headline PPL/KLD (code/math/tools) — disjoint from every cal source",
    }

    gtext, gn, grows = external.sample_parquet_text(
        external.download_parquet(external.GENERAL_EVAL_DOMAIN), tok,
        target_tokens=cfg.general_eval_tokens, seed=cfg.seed + len(external.EVAL_DOMAINS),
    )
    (out / "corpus.eval.general.txt").write_text(gtext)
    audit["eval"]["general"] = {
        "path": str(out / "corpus.eval.general.txt"),
        "domain": external.GENERAL_EVAL_DOMAIN, "tokens": gn, "rows": grows,
    }

    if cfg.enabled(SOURCE_LOGS):
        tchunks, _k, ttot, tpack = split.stratified_pack(
            log_splits["holdout"], tok, target_tokens=cfg.tools_eval_tokens, **pack_kwargs,
        )
        split.write_corpus(tchunks, out / "corpus.eval.tools.txt")
        audit["eval"]["tools"] = {
            "path": str(out / "corpus.eval.tools.txt"), "source_slice": "logs.holdout (cli + agents)",
            "tokens": ttot, "windows": len(tchunks), "pack_audit": tpack,
            "caveat": "llama-perplexity has no --parse-special: chat markers tokenize as "
                      "plain BPE. Quant-vs-quant only, not absolute PPL.",
        }

    if swe_eval:
        achunks, _k, atot, apack = split.stratified_pack(
            swe_eval, tok, target_tokens=cfg.agentic_eval_tokens, **pack_kwargs,
        )
        split.write_corpus(achunks, out / "corpus.eval.agentic.txt")
        audit["eval"]["agentic"] = {
            "path": str(out / "corpus.eval.agentic.txt"),
            "source": f"{SWE_DATASET}:{SWE_SPLIT} holdout",
            "tokens": atot, "windows": len(achunks), "pack_audit": apack,
            "caveat": "same --parse-special caveat as the tools eval",
        }

    if broad_eval:
        bchunks, btot = pack_raw_samples(
            broad_eval, tok, cfg.broad_eval_tokens, cfg.window_cap, cfg.seed + 2,
        )
        split.write_corpus(bchunks, out / "corpus.eval.broad.txt")
        audit["eval"]["broad"] = {
            "path": str(out / "corpus.eval.broad.txt"),
            "source": f"{BROAD_DATASET}:{BROAD_SPLIT} (mtp half, eval slice)",
            "tokens": btot, "chunks": len(bchunks),
            "used_for": "did low-bit quantization cost general knowledge outside the "
                        "coding/tool distribution the imatrix was fit on?",
        }

    if redteam_eval:
        rchunks, _k, rtot, rpack = split.stratified_pack(
            redteam_eval, tok, target_tokens=cfg.redteam_eval_tokens, **pack_kwargs,
        )
        split.write_corpus(rchunks, out / "corpus.eval.redteam.txt")
        audit["eval"]["redteam"] = {
            "path": str(out / "corpus.eval.redteam.txt"),
            "source": f"{refusals.REDTEAM_DATASET}:{refusals.REDTEAM_SPLIT} holdout, refused",
            "tokens": rtot, "windows": len(rchunks), "pack_audit": rpack,
            "used_for": "held-out attack prompts paired with refusals. PPL here is a proxy "
                        "only — actual refusal behavior is measured by scripts/"
                        "eval_redteam.py against the built quant, not by this corpus.",
        }

    # --- 8) disjointness, asserted ----------------------------------------------------
    if cfg.enabled(SOURCE_LOGS):
        tr, te, ho = ({fp(s) for s in log_splits[k]} for k in ("train", "test", "holdout"))
        assert not (tr & te), "logs train ∩ test non-empty"
        assert not (tr & ho), "logs train ∩ holdout non-empty"
        assert not (te & ho), "logs test ∩ holdout non-empty"
    if swe_cal or swe_eval:
        cal_ids = {s["id"] for s in swe_cal}
        assert not (cal_ids & {s["id"] for s in swe_eval}), "swe cal ∩ eval non-empty"
    if broad_cal:
        cal_h = {hashlib.sha256(t.encode()).hexdigest() for t in broad_cal}
        ev_h = {hashlib.sha256(t.encode()).hexdigest() for t in broad_eval + broad_val}
        assert not (cal_h & ev_h), "broad cal ∩ (eval ∪ val) non-empty"
    if redteam_cal or redteam_eval:
        assert not ({s["id"] for s in redteam_cal} & {s["id"] for s in redteam_eval}), \
            "redteam cal ∩ eval non-empty"
        # The disclosure rows carry the target's original (sometimes harmful) completion.
        # Nothing but the attack turns and our refusals may reach a corpus — check the
        # actual bytes rather than trusting the transform.
        leaked = [s["id"] for s in redteam_cal + redteam_eval
                  for m in s["messages"]
                  if m["role"] == "assistant"
                  and m["content"] not in refusals.GENERIC_REFUSALS]
        assert not leaked, f"non-refusal assistant content in redteam sessions: {leaked[:3]}"
    audit["disjointness"] = ("asserted: logs train/test/holdout (grouped by issue), "
                         "swe cal/eval, broad cal/eval")
    print("  disjointness: OK", file=sys.stderr)

    # --- 9) the SFT export: full conversations, nothing trimmed -----------------------
    if cfg.write_sft:
        audit["sft"] = _write_sft(
            cfg, out, tok,
            log_splits=log_splits,
            swe_cal=swe_cal, swe_eval=swe_eval,
            redteam_cal=redteam_cal, redteam_eval=redteam_eval,
        )

    (out / "corpora_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(f"  audit:       {out / 'corpora_audit.json'}", file=sys.stderr)
    return audit


__all__ = [
    "ALL_SOURCES",
    "BROAD_DATASET",
    "SOURCE_BROAD",
    "SOURCE_LOGS",
    "SOURCE_REASONING",
    "SOURCE_REDTEAM",
    "SOURCE_SWE",
    "SOURCE_WIKI",
    "SWE_DATASET",
    "UniversalConfig",
    "build",
    "clip_tool_messages",
    "pack_raw_samples",
    "reasoning_windows",
    "sampled_coverage",
    "scan_special_tokens",
    "sft_record",
    "swe_sessions",
    "tool_call_marker_counts",
    "write_jsonl_gz",
]
