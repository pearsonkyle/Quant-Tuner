"""P(verbatim repeat) measured in a RECONSTRUCTED real episode state.

The constructed bank contexts and the real agent episode can disagree (anchor9:
0.08 flat on held-out bank contexts, yet a 29x loop in the mimic). This rebuilds the
episode from its recorded trajectory — the actual commands, the actual output heads,
the actual depth — and measures P(re-issuing the looped command) at chosen positions
along the streak. If P is high here while the bank reads low, the residual pathology
lives in self-generated long-context state, and the next lever is on-policy training,
not more constructed contexts.

    PYTHONPATH=src .venv/bin/python scripts/measure_traj_repeat.py \
        --traj /workspace/swe-mimic/work/dask__dask-11393/traj_ANCHOR9-Q2_0.json \
        --instance /workspace/swe-mimic/instance.json \
        --latents out/exp-058/kd32b-full-anchor9/trained_latents.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.steer import TOOLS  # noqa: E402

SYSTEM_TMPL = """You are a software engineer fixing a bug in a Python repository.

Your shell starts in the repository root, which is `{repo}`. There is NO /testbed directory
on this machine — use paths relative to the repository root. A test has been added that
currently FAILS; your job is to
change the SOURCE code so it passes, without breaking existing tests.

Use the `bash` tool to explore, edit, and run tests. Work incrementally:
  1. Find the relevant source file (grep/find).
  2. Read the failing test to understand exactly what is expected.
  3. Make a minimal, correct edit to the source.
  4. Re-run the failing test to confirm.

Edit files with `python - <<'EOF' ... EOF` heredocs or `sed`. Do NOT edit test files.
When the failing test passes, say DONE."""


def call_text(cmd: str) -> str:
    args = json.dumps({"command": cmd})
    return f'<tool_call>\n{{"name": "bash", "arguments": {args}}}\n</tool_call>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--latents", default=None)
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="truncate the episode prefix to its LAST N tokens (keeping the "
                         "system+task head) — measures whether a shortened real prefix "
                         "still elicits the loop, which sets the training-context cost")
    ap.add_argument("--probe-at", default="1,2,3,5,10,15,20,25,29",
                    help="streak positions (occurrence count of the looped command)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    traj = json.loads(Path(args.traj).read_text())
    inst = json.loads(Path(args.instance).read_text())

    # locate the longest identical streak
    best, streak = (1, 1), 1
    for i in range(1, len(traj)):
        if traj[i]["cmd"] == traj[i - 1]["cmd"]:
            streak += 1
            if streak > best[0]:
                best = (streak, i - streak + 1)
        else:
            streak = 1
    n_streak, s0 = best
    loop_cmd = traj[s0]["cmd"]
    print(f"[traj] {len(traj)} steps; streak {n_streak}x from step {s0}: "
          f"{loop_cmd[:80]!r}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant_tuner.qat.attention import enable_fp32_gqa_repeat
    from quant_tuner.qat.train import wrap_model

    if args.device == "cuda":
        enable_fp32_gqa_repeat()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32).to(args.device)
    model.eval()
    wrap_model(model, 36)
    if args.latents:
        sd = torch.load(args.latents, map_location="cpu", weights_only=False, mmap=True)
        latents = sd["latents"]
        n = 0
        with torch.no_grad():
            for name_, mod in model.named_modules():
                key = f"{name_}.linear.weight"
                src = latents.get(key, latents.get(key.replace(".linear.weight", ".weight")))
                if src is not None and hasattr(mod, "linear"):
                    mod.linear.weight.copy_(src.to(args.device))
                    n += 1
        print(f"[traj] loaded {n} latent tensors", flush=True)

    repo = inst["repo"].split("/")[-1] if "/" in inst.get("repo", "") else inst.get("repo", "repo")
    prefix = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_TMPL.format(repo=repo)},
         {"role": "user", "content": inst["problem_statement"][:4000]}],
        tools=TOOLS, tokenize=False, add_generation_prompt=True)

    forced = call_text(loop_cmd)
    forced_ids = tok(forced, add_special_tokens=False, return_tensors="pt").input_ids[0]
    probe_at = [int(x) for x in args.probe_at.split(",") if int(x) <= n_streak]
    for occ in probe_at:
        # context = everything up to (but not including) the occ-th streak repeat
        upto = s0 + occ - 1
        ctx = prefix
        for e in traj[:upto]:
            out = (e.get("out_head") or "").strip() or "(no output)"
            ctx += (call_text(e["cmd"]) + "<|im_end|>\n<|im_start|>user\n"
                    "<tool_response>\n" + out + "\n</tool_response><|im_end|>\n"
                    "<|im_start|>assistant\n")
        ids = tok(ctx, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if args.max_ctx and ids.shape[0] > args.max_ctx:
            head = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
            tail_len = args.max_ctx - head.shape[0]
            ids = torch.cat([head, ids[-tail_len:]])
        row = torch.cat([ids, forced_ids]).unsqueeze(0).to(args.device)
        lo = ids.shape[0]
        with torch.no_grad():
            h = model.model(input_ids=row).last_hidden_state
            lg = model.lm_head(h[0, lo - 1:row.shape[1] - 1]).float()
        lp = torch.log_softmax(lg, -1).gather(
            -1, forced_ids.to(args.device).unsqueeze(-1)).squeeze(-1)
        print(f"[traj] after {occ:>2} identical rounds (ctx {row.shape[1]:>6} tok): "
              f"P(repeat) = {lp.mean().exp().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
