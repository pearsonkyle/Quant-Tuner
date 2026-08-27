#!/usr/bin/env python3
"""What did the agent actually DO — and did it fail in a way that means something?

    python scripts/analyze_swe_anomalies.py --label SFT32K_SW1-Q2_0
    python scripts/analyze_swe_anomalies.py --all --print

`resolved=0` is the same number for a model that never emitted a token, one that looped on
`ls` for sixty turns, and one that read the code, wrote a patch and got the fix subtly
wrong. Those are three different problems and only the last is about capability, so a
pass-rate column alone cannot steer training. This reads the saved trajectory and names
the failure mode.

The modes it separates, and what each implies:

* **mute** — near-zero output tokens, no tool calls. The model emitted its stop token
  immediately. This is a TERMINATION failure, not a capability one, and it is what
  `--stop-weight 6.0` produced: read it next to `stop_prob.csv`, where the same model
  shows P(stop) 0.97 after a sentence.
* **loop** — the same command repeated. The opposite termination failure: the model never
  stops. This is what an under-weighted stop signal produces.
* **flailing** — many commands, high malformed rate: the model is trying but cannot
  produce well-formed tool calls.
* **worked, unresolved** — a real attempt that did not fix the bug. The only mode that is
  genuinely about capability, and the only one where more/better data is the answer.

Reads only; it never launches a server or an agent.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

MIMIC = Path("/workspace/swe-mimic")
WORK = MIMIC / "work"

# A model that emits fewer than this many output tokens across a whole episode has not
# attempted the task. The vanilla control produced 1,781; the mute sft32k produced 1.
MUTE_TOKENS = 50
# Repeating one command this many times in a row is not exploration.
LOOP_RUN = 3


def _traj_dir() -> Path:
    cands = [p for p in WORK.glob("*") if p.is_dir()]
    if not cands:
        raise SystemExit(f"no instance dir under {WORK}")
    return cands[0]


def longest_repeat_run(cmds: list[str]) -> tuple[int, str | None]:
    best, best_cmd, run, prev = 0, None, 0, None
    for c in cmds:
        run = run + 1 if c == prev else 1
        if run > best:
            best, best_cmd = run, c
        prev = c
    return best, best_cmd


def degenerate_text(s: str, min_len: int = 24) -> bool:
    """A short substring repeated to fill the buffer — the classic decode collapse."""
    s = s.strip()
    # min_len, not a multiple of it: the regex below already requires 24+ characters of
    # actual repetition, and a stricter length gate here made "abab..."-style collapse
    # invisible in anything short of a paragraph.
    if len(s) < min_len:
        return False
    for n in (1, 2, 3, 4, 8, 16):
        unit = s[:n]
        if unit and s == (unit * (len(s) // n + 1))[:len(s)]:
            return True
    # also catch a repeated token-ish chunk anywhere
    return bool(re.search(r"(.{4,40}?)\1{5,}", s))


def analyze(label: str, tdir: Path) -> dict:
    res_p = tdir / f"result_{label}.json"
    traj_p = tdir / f"traj_{label}.json"
    if not res_p.exists():
        return {"label": label, "error": f"no result_{label}.json"}
    res = json.loads(res_p.read_text())
    traj = json.loads(traj_p.read_text()) if traj_p.exists() else []

    cmds = [t.get("cmd", "") for t in traj]
    kinds = collections.Counter(t.get("kind", "?") for t in traj)
    run_len, run_cmd = longest_repeat_run(cmds)
    distinct = len(set(cmds))
    out_tok = int(res.get("out_tokens") or 0)
    n_calls = len(traj)

    degen = [i for i, t in enumerate(traj) if degenerate_text(t.get("cmd", ""))]

    flags = []
    if out_tok <= MUTE_TOKENS and n_calls == 0:
        mode = "mute"
        flags.append(f"{out_tok} output tokens, 0 tool calls — emitted its stop token "
                     f"immediately; a TERMINATION failure, not a capability one")
    elif run_len >= LOOP_RUN:
        mode = "loop"
        flags.append(f"repeated the same command {run_len}x in a row: {run_cmd!r}")
    elif n_calls and distinct / n_calls < 0.4:
        mode = "loop"
        flags.append(f"only {distinct} distinct commands in {n_calls} calls "
                     f"({distinct / n_calls:.0%} distinct)")
    elif n_calls and (kinds.get("malformed", 0) / n_calls) > 0.3:
        mode = "flailing"
        flags.append(f"{kinds['malformed']}/{n_calls} commands malformed")
    elif res.get("resolved"):
        mode = "resolved"
    elif res.get("patch_produced"):
        mode = "patched, unresolved"
    else:
        mode = "worked, unresolved"

    if res.get("hit_max_turns"):
        flags.append("hit max turns — never terminated on its own")
    if degen:
        flags.append(f"degenerate repeated text in {len(degen)} command(s)")
    if n_calls and kinds.get("timeout"):
        flags.append(f"{kinds['timeout']} command timeout(s)")
    if out_tok and n_calls:
        per = out_tok / n_calls
        if per < 20:
            flags.append(f"{per:.0f} output tokens per tool call — unusually terse")

    return {
        "label": label,
        "mode": mode,
        "flags": flags,
        "resolved": res.get("resolved"),
        "patch_produced": res.get("patch_produced"),
        "n_tool_calls": n_calls,
        "distinct_commands": distinct,
        "longest_repeat_run": run_len,
        "repeated_command": run_cmd if run_len >= LOOP_RUN else None,
        "kinds": dict(kinds),
        "out_tokens": out_tok,
        "in_tokens": res.get("in_tokens"),
        "tool_err_rate": res.get("tool_err_rate"),
        "exit_status": res.get("exit_status"),
        "hit_max_turns": res.get("hit_max_turns"),
        "wall_s": res.get("wall_s"),
        "f2p": f"{res.get('f2p_passed')}/{res.get('f2p_total')}",
        "p2p": f"{res.get('p2p_passed')}/{res.get('p2p_total')}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="one label, e.g. SFT32K_SW1-Q2_0")
    ap.add_argument("--all", action="store_true", help="every label present")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--print", dest="do_print", action="store_true")
    a = ap.parse_args()

    tdir = _traj_dir()
    if a.all or not a.label:
        labels = sorted(p.name[len("result_"):-len(".json")]
                        for p in tdir.glob("result_*.json"))
    else:
        labels = [a.label]

    reports = [analyze(lbl, tdir) for lbl in labels]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(reports if len(reports) > 1 else reports[0],
                                    indent=2) + "\n")
        print(f"[anomaly] -> {a.out}")

    for r in reports:
        if r.get("error"):
            print(f"  {r['label']}: {r['error']}")
            continue
        print(f"  {r['label']:22s} {r['mode']:20s} "
              f"calls={r['n_tool_calls']:<3} out_tok={r['out_tokens']:<7} "
              f"f2p={r['f2p']} p2p={r['p2p']}")
        for f in r["flags"]:
            print(f"      ! {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
