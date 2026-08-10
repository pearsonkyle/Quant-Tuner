"""Does this model's chat template render TOOL CALLS the way the corpus builder assumes?

Run this FIRST for any new model — before extraction, before corpora, before an imatrix.
It loads only the tokenizer, so it takes seconds and needs no GPU or GGUF.

    uv run python scripts/verify_chat_template.py --model Qwen/Qwen3.8-27B
    uv run python scripts/verify_chat_template.py --model out/exp-060/model_extracted --show

Exit code is non-zero when a blocking check fails (see quant_tuner.data.template_check for
what is blocking vs. a warning).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data.template_check import check_template


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF repo id or local dir carrying tokenizer_config.json")
    p.add_argument("--show", action="store_true",
                   help="print the full rendered tool-calling fixture")
    p.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    a = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=True)
    rep = check_template(tok, a.model)

    if a.show:
        print("=" * 78)
        print(rep.render)
        print("=" * 78)
    print(rep.summary())
    if a.json:
        a.json.write_text(json.dumps(rep.to_dict(), indent=2))
        print(f"  wrote {a.json}")

    if not rep.ok:
        print("\nThis template cannot carry tool calls into a calibration corpus. Fix the "
              "template (or the tokenizer files) before building corpora — every "
              "downstream number depends on it.", file=sys.stderr)
        return 1
    if rep.warnings:
        print("\nWarnings above are informational: most often an unrecognised marker "
              "family. Eyeball the render with --show and, if the calls ARE structured, "
              "add the marker to template_check.KNOWN_TOOL_CALL_MARKERS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
