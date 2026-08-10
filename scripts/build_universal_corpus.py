"""Build the universal calibration corpus: every dataset in datasets/, plus raw wiki.

The successor to ``scripts/build_corpora.py``. Prefer this for any new model;
the older builder stays for reproducing published two-source runs.

    uv run python scripts/build_universal_corpus.py \\
        --out out/exp-060/corpora \\
        --model out/exp-060/model_extracted \\
        --wiki out/exp-001/wiki/wiki.test.raw

Writes into ``--out``:

    corpus.cal.txt            the calibration corpus (all sources interleaved)
    corpus.cal.<source>.txt   per-source intermediates, for eyeballing
                              (incl. corpus.cal.reasoning.txt — windows cut so a reasoning
                               turn lands last, the only position a template keeps it in)
    corpus.val.txt            AWQ cv-scoring slice (in-domain logs + out-of-domain breadth)
    corpus.eval.txt           external code/math/tools     -> headline PPL/KLD
    corpus.eval.general.txt   external broad English        -> its own baseline.kld
    corpus.eval.tools.txt     on-disk logs holdout          -> its own baseline.kld
    corpus.eval.agentic.txt   SWE trajectory holdout        -> its own baseline.kld
    corpus.eval.broad.txt     broad-supplement holdout      -> its own baseline.kld
    corpus.eval.redteam.txt   held-out attacks + refusals   -> its own baseline.kld
    corpora_audit.json        token counts, per-source shares, tool-call marker scan,
                              and the chat-template report

Each eval corpus is a SEPARATE distribution and must get its own ``kld.build_baseline``;
do not concatenate them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import universal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="HF dir with the tokenizer + chat template "
                        "(e.g. out/<run>/model_extracted)")
    p.add_argument("--wiki", type=Path, default=universal.DEFAULT_WIKI)
    p.add_argument("--logs", type=Path, nargs="+", default=list(universal.DEFAULT_LOG_FILES),
                   help="on-disk log JSONL(.gz) files (default: datasets/agent-logs/data/"
                        "logs-cli.jsonl.gz + logs-agents.jsonl.gz)")
    p.add_argument("--broad-jsonl", type=Path, default=None,
                   help=f"local override for {universal.BROAD_DATASET} (default: the "
                        "staged datasets/ copy, else the Hub)")
    p.add_argument("--swe-jsonl", type=Path, default=None,
                   help=f"local override for {universal.SWE_DATASET}")
    p.add_argument("--sources", nargs="+",
                   default=list(universal.UniversalConfig.sources),
                   choices=list(universal.ALL_SOURCES),
                   help="subset of sources to include (default: all of them)")
    p.add_argument("--cal-logs-tokens", type=int, default=2_000_000,
                   help="budget for BOTH on-disk log corpora combined")
    p.add_argument("--cal-swe-tokens", type=int, default=1_000_000)
    p.add_argument("--cal-broad-tokens", type=int, default=None,
                   help="omit to use ALL of the supplement's calib half (the default)")
    p.add_argument("--cal-reasoning-tokens", type=int, default=1_000_000,
                   help="budget for reasoning-terminal windows — the only way reasoning "
                        "reaches the corpus at all (0 disables)")
    p.add_argument("--cal-redteam-tokens", type=int, default=None,
                   help="budget for the refusal source (attack prompts + generic "
                        "refusals). Omit to use all of them (the default)")
    p.add_argument("--cal-wiki-tokens", type=int, default=None,
                   help="cap wiki's contribution (default: all of it). Check "
                        "token_share in the audit — with small chat budgets wiki "
                        "otherwise dominates the mix")
    p.add_argument("--eval-tokens-per-domain", type=int, default=30_000)
    p.add_argument("--per-session-cap", type=int, default=3_500,
                   help="max tokens per window; keep < the imatrix ctx (default 4096)")
    p.add_argument("--max-tool-output-tokens", type=int, default=512,
                   help="head+tail clip for role=tool contents (0 disables)")
    p.add_argument("--reasoning", default="auto", choices=["auto", "field", "drop"],
                   help="how assistant reasoning is normalized before templating; "
                        "'drop' for a non-thinking target model (default auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-template-warnings", action="store_true",
                   help="build even if the chat-template check FAILS (it always warns "
                        "loudly; use only after inspecting the render by hand)")
    p.add_argument("--allow-no-tool-calls", action="store_true",
                   help="do not fail when the built corpus has zero tool-call markers")
    a = p.parse_args()

    cfg = universal.UniversalConfig(
        out_dir=a.out,
        model_dir=a.model,
        log_files=tuple(a.logs),
        wiki=a.wiki,
        broad_jsonl=a.broad_jsonl,
        swe_jsonl=a.swe_jsonl,
        sources=tuple(a.sources),
        cal_logs_tokens=a.cal_logs_tokens,
        cal_swe_tokens=a.cal_swe_tokens,
        cal_broad_tokens=a.cal_broad_tokens,
        cal_redteam_tokens=a.cal_redteam_tokens,
        cal_reasoning_tokens=a.cal_reasoning_tokens,
        reasoning_policy=a.reasoning,
        cal_wiki_tokens=a.cal_wiki_tokens,
        eval_tokens_per_domain=a.eval_tokens_per_domain,
        per_session_cap=a.per_session_cap,
        max_tool_output_tokens=a.max_tool_output_tokens,
        seed=a.seed,
        strict_template=not a.allow_template_warnings,
        require_tool_calls=not a.allow_no_tool_calls,
    )
    audit = universal.build(cfg)
    share = audit["calibration"]["token_share"]
    print("\n=== universal corpus built ===")
    print(f"  calibration tokens: {audit['calibration']['total_tokens']:,}")
    print(f"  source shares:      {share}")
    print(f"  tool-call markers:  {audit['calibration']['tool_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
