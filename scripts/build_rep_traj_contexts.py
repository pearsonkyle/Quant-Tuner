"""Harvest loop-state training contexts from recorded agent trajectories.

anchor9 proved repetition suppression is state-dependent: 0.08-0.15 on constructed
contexts at any depth, 0.96+ in the reconstructed real episode. The trainable states
must therefore BE (near-)real episode states. This scans harvest trajectories
(gen_harvest.sh episodes on training-pool instances — the eval instance is refused),
finds every decision point where the next command verbatim-repeats an earlier one,
and emits JSONL rows {ctx, rep} rendered exactly like measure_traj_repeat.py:
system + task + the real (cmd, out_head) history, teacher-forced repeat appended by
the training loss. Contexts are optionally truncated (head + tail) to a token budget.

    PYTHONPATH=src .venv/bin/python scripts/build_rep_traj_contexts.py \
        --work /workspace/swe-mimic/work --label HARVEST-anchor9 \
        --out out/exp-058/kd/rep_traj_contexts.jsonl --max-ctx 3072
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from measure_traj_repeat import SYSTEM_TMPL, call_text  # noqa: E402

from quant_tuner.qat.steer import TOOLS  # noqa: E402
from quant_tuner.qat.stop_probe import SENTENCE as PROBE_SENTENCE  # noqa: E402
from quant_tuner.qat.stop_probe import USER as PROBE_USER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="mimic work/ dir holding <iid>/")
    ap.add_argument("--label", required=True, help="trajectory label (traj_<label>.json)")
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ctx", type=int, default=8192,
                    help="hard cap only — MEASURED: truncating the real prefix to 3k "
                         "collapses P(repeat) 0.96->0.06; the loop state needs its "
                         "full history, so keep contexts whole")
    ap.add_argument("--min-occurrence", type=int, default=1,
                    help="harvest from the Nth verbatim re-issue on (1 = first repeat)")
    ap.add_argument("--max-per-episode", type=int, default=6,
                    help="cap decision points per episode (spread over the streak)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    rows = []
    for inst_dir in sorted(Path(args.work).iterdir()):
        traj_p = inst_dir / f"traj_{args.label}.json"
        inst_p = inst_dir / "instance.json"
        if not traj_p.exists() or not inst_p.exists():
            continue
        if inst_dir.name.startswith("dask__"):
            raise SystemExit(f"refusing eval instance {inst_dir.name} in the harvest")
        traj = json.loads(traj_p.read_text())
        inst = json.loads(inst_p.read_text())
        repo = inst["repo"].split("/")[-1] if "/" in inst.get("repo", "") else "repo"
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_TMPL.format(repo=repo)},
             {"role": "user", "content": inst["problem_statement"][:4000]}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True)
        # decision points: step i whose cmd already appeared >= min_occurrence times
        seen: dict[str, int] = {}
        points = []
        for i, e in enumerate(traj):
            k = seen.get(e["cmd"], 0)
            if k >= args.min_occurrence:
                points.append((i, k))
            seen[e["cmd"]] = k + 1
        if not points:
            continue
        stride = max(1, len(points) // args.max_per_episode)
        for i, occ in points[::stride][:args.max_per_episode]:
            ctx = prefix
            for e in traj[:i]:
                out = (e.get("out_head") or "").strip() or "(no output)"
                ctx += (call_text(e["cmd"]) + "<|im_end|>\n<|im_start|>user\n"
                        "<tool_response>\n" + out + "\n</tool_response><|im_end|>\n"
                        "<|im_start|>assistant\n")
            rep = call_text(traj[i]["cmd"])
            assert PROBE_SENTENCE not in ctx and PROBE_USER not in ctx
            ids = tok(ctx, add_special_tokens=False).input_ids
            if len(ids) > args.max_ctx:
                head = tok(prefix, add_special_tokens=False).input_ids
                tail = ids[-(args.max_ctx - len(head)):]
                ctx = tok.decode(head + tail)
            rows.append({"ctx": ctx, "rep": rep, "instance": inst_dir.name,
                         "step": i, "occurrence": occ})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[r["instance"]] = by.get(r["instance"], 0) + 1
    print(f"[harvest] {len(rows)} contexts -> {out}  per-instance: {by}", flush=True)


if __name__ == "__main__":
    main()
