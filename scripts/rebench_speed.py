"""Re-run llama-bench on every GGUF, in series, with 10 repetitions.

Runs strictly sequentially so models don't compete for GPU memory. Replaces
the speed columns (prefill/decode/TTFT + new stdev columns) in results.csv
without touching KLD/PPL/BPW.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import speed
from quant_tuner.bench.runner import CSV_COLUMNS

WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
RESULTS = WORK / "results.csv"

# (label_in_results_csv, gguf_path)
MODELS = [
    ("baseline/fp16",         WORK / "model-f16.gguf"),
    ("baseline/Q4_K_M-none",  WORK / "Q4_K_M-none.gguf"),
    ("stock/Q4_K_M-custom",   WORK / "Q4_K_M-stock_custom.gguf"),
    ("stock/Q4_K_M-wiki",     WORK / "Q4_K_M-stock_wiki.gguf"),
    ("stock/Q4_K_M-mixed",    WORK / "Q4_K_M-stock_mixed.gguf"),
    ("hybrid/Q4_K_M-custom",  WORK / "Q4_K_M-hybrid_custom.gguf"),
    ("hybrid/Q4_K_M-wiki",    WORK / "Q4_K_M-hybrid_wiki.gguf"),
    ("hybrid/Q4_K_M-mixed",   WORK / "Q4_K_M-hybrid_mixed.gguf"),
]

REPETITIONS = 10
NEW_SPEED_FIELDS = (
    "prefill_tok_s", "prefill_stdev",
    "decode_tok_s", "decode_stdev",
    "ttft_2k_ms", "ttft_stdev_ms",
    "bench_repetitions",
)


def load_rows(path: Path) -> tuple[list[str], dict[str, dict]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        by_label = {r["model"]: r for r in reader}
    return fields, by_label


def write_rows(path: Path, fields: list[str], by_label: dict[str, dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in by_label.values():
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    assert RESULTS.exists(), f"missing results.csv: {RESULTS}"
    fields, by_label = load_rows(RESULTS)
    # Ensure the new stdev fields are present in the CSV schema.
    for f in CSV_COLUMNS:
        if f not in fields:
            fields.append(f)

    for label, model in MODELS:
        if not model.exists():
            print(f"[skip] missing {model.name}", flush=True)
            continue
        if label not in by_label:
            print(f"[skip] {label!r} not in results.csv (run kld bench first)", flush=True)
            continue
        print(f"\n==> bench {label}  ({model.name})", flush=True)
        t0 = time.time()
        m = speed.evaluate(
            model,
            repetitions=REPETITIONS,
            log=LOGS / f"speed_{model.stem}.log",
        )
        dt = (time.time() - t0) / 60
        prefill = m.prefill_tok_s
        decode = m.decode_tok_s
        ttft = m.ttft_2k_ms
        p_sd = m.prefill_stdev
        d_sd = m.decode_stdev
        t_sd = m.ttft_stdev_ms
        print(
            f"    {dt:.1f} min   "
            f"prefill {prefill:.2f} ± {p_sd if p_sd is not None else math.nan:.2f}   "
            f"decode {decode:.2f} ± {d_sd if d_sd is not None else math.nan:.2f}   "
            f"ttft {ttft:.1f} ± {t_sd if t_sd is not None else math.nan:.1f} ms",
            flush=True,
        )
        row = by_label[label]
        for k in NEW_SPEED_FIELDS:
            v = getattr(m, "n_repetitions" if k == "bench_repetitions" else k)
            row[k] = "" if v is None else v
        write_rows(RESULTS, fields, by_label)  # snapshot per-model

    print("\nALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
