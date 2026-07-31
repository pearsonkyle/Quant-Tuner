"""exp-045 experiment: graft Qwopus3.6's trained MTP (nextn) head onto tmax-27b.

tmax-27b and Jackrong/Qwopus3.6-27B-Coder are both full finetunes of the SAME
Qwen3.6-27B base, with byte-identical architecture (hidden 5120, 64 layers, 24/4
GQA, head_dim 256, vocab 248320, same hybrid linear/full-attn layout). tmax shipped
WITHOUT its MTP head; Qwopus kept+trained one (15 `mtp.*` tensors: an `fc` fusion
proj, one transformer block, and norms; it shares the trunk's embed_tokens/lm_head).

Because the dims match exactly, we can graft Qwopus's `mtp.*` weights onto tmax's
trunk and flip `mtp_num_hidden_layers` 0 -> 1, producing a GGUF whose blk.64 is a
draftable nextn head. Whether it drafts WELL (acceptance) is empirical — the head was
trained on Qwopus's (Coder) hidden states, not tmax's (DPPO) — so bench acceptance
afterward (bench_mtp_speed.py, n-max 1..4).

Mechanics: we don't touch tmax's 50GB trunk shards — we symlink them into a new HF
dir and add ONE extra shard (`model-mtp.safetensors`) holding the 15 grafted tensors,
then merge the safetensors index and patch the config.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp045_graft_mtp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
TMAX_HF = REPO / "out" / "exp-045" / "model_extracted"
GRAFT_HF = REPO / "out" / "exp-045" / "model_extracted_mtp"


def _qwopus_snapshot() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = sorted(hub.glob("*Qwopus3.6-27B-Coder*/snapshots/*"))
    if not snaps:
        raise FileNotFoundError("Qwopus3.6-27B-Coder not in HF cache")
    return snaps[0]


def main() -> int:
    if not (TMAX_HF / "config.json").exists():
        raise FileNotFoundError(f"run exp045_setup_tmax.py first: {TMAX_HF}")
    qsnap = _qwopus_snapshot()
    GRAFT_HF.mkdir(parents=True, exist_ok=True)

    # 1. symlink tmax trunk shards + tokenizer/aux files into the graft dir
    for item in TMAX_HF.iterdir():
        if item.name in ("config.json", "model.safetensors.index.json"):
            continue
        dst = GRAFT_HF / item.name
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(item.resolve())

    # 2. load Qwopus's mtp.* tensors and write them as one new shard
    qidx = json.loads((qsnap / "model.safetensors.index.json").read_text())["weight_map"]
    mtp_keys = sorted(k for k in qidx if k.startswith("mtp."))
    print(f"grafting {len(mtp_keys)} mtp.* tensors from {qsnap.name}")
    handles: dict[str, object] = {}
    mtp_tensors: dict[str, torch.Tensor] = {}
    for k in mtp_keys:
        sf = qidx[k]
        if sf not in handles:
            handles[sf] = safe_open(qsnap / sf, framework="pt")
        mtp_tensors[k] = handles[sf].get_tensor(k)
    mtp_shard = "model-mtp.safetensors"
    save_file(mtp_tensors, str(GRAFT_HF / mtp_shard), metadata={"format": "pt"})
    mtp_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"  wrote {mtp_shard} ({mtp_bytes/1024**2:.0f} MiB)")

    # 3. merged safetensors index = tmax trunk weight_map + the grafted mtp keys
    tidx = json.loads((TMAX_HF / "model.safetensors.index.json").read_text())
    weight_map = dict(tidx["weight_map"])
    for k in mtp_keys:
        weight_map[k] = mtp_shard
    total = int(tidx.get("metadata", {}).get("total_size", 0)) + mtp_bytes
    (GRAFT_HF / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    # 4. config: tmax's, but re-enable the MTP layer so the converter emits blk.64
    cfg = json.loads((TMAX_HF / "config.json").read_text())
    cfg["mtp_num_hidden_layers"] = 1
    if "num_nextn_predict_layers" in cfg:
        cfg["num_nextn_predict_layers"] = 1
    (GRAFT_HF / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nDONE → {GRAFT_HF}")
    print("  config mtp_num_hidden_layers = 1 (grafted Qwopus nextn head)")
    print("  Next: convert -> F16 (expect blk.64), quantize IQ4_XS pinning blk.64@Q8,")
    print("  then bench_mtp_speed.py --n-max {1,2,3,4} for acceptance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
