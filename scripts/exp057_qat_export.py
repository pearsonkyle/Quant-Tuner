"""exp-057 export: trained ternary latents -> Q2_0 GGUF (runnable on the prism fork).

Loads the unpacked base model, overwrites the trained layers' weights with the
QAT latents, then ternarizes EVERY wrapped linear with the SAME TWN quantizer the
forward used (frozen layers = no-op; trained layers = their final ternary codes).
The result is a pure-ternary HF model -> F16 GGUF -> Q2_0 (lossless, weights are
already on the grid). Q2_0 output needs the prism llama-quantize (type 41).

    LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
        scripts/exp057_qat_export.py --latents out/exp-057/trained_scope/trained_latents.pt \
        --tag scope
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.qat.ternary import ternarize_group
from quant_tuner.quantize import convert

MODEL = REPO / "out" / "exp-057" / "model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--tag", default="qat")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_hf = REPO / "out" / "exp-057" / f"model_{args.tag}"
    f16 = REPO / "out" / "exp-057" / f"Ternary-Bonsai-8B-{args.tag}-F16.gguf"
    q2 = REPO / "out" / "exp-057" / f"Ternary-Bonsai-8B-{args.tag}-Q2_0.gguf"

    print(f"[export] loading base {MODEL} (fp32, cpu)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

    # 1) overwrite trained-layer weights with the QAT latents
    blob = torch.load(args.latents, map_location="cpu", weights_only=False)
    latents = blob["latents"]
    sd = dict(model.named_parameters())
    n_over = 0
    for k, v in latents.items():
        base_key = k.replace(".linear.weight", ".weight")
        if base_key in sd:
            sd[base_key].data.copy_(v.to(torch.float32))
            n_over += 1
    print(f"[export] overwrote {n_over}/{len(latents)} trained tensors "
          f"(loss {blob.get('loss_first'):.3f}->{blob.get('loss_last'):.3f})", flush=True)

    # 2) ternarize every qualifying linear in the decoder (frozen = no-op)
    n_tern = 0
    with torch.no_grad():
        for layer in model.model.layers:
            for mod in layer.modules():
                if isinstance(mod, torch.nn.Linear) and mod.in_features % 128 == 0:
                    _, _, w_hat = ternarize_group(mod.weight.data)
                    mod.weight.data.copy_(w_hat)
                    n_tern += 1
    print(f"[export] ternarized {n_tern} linears", flush=True)

    out_hf.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_hf)
    tok.save_pretrained(out_hf)
    print(f"[export] saved HF -> {out_hf}", flush=True)

    # 3) F16 GGUF, then Q2_0 (prism llama-quantize)
    print("[export] convert -> F16 GGUF ...", flush=True)
    convert.hf_to_f16_gguf(out_hf, f16, log=REPO / "out" / "exp-057" / f"convert_{args.tag}.log")

    from quant_tuner import paths
    import subprocess
    qbin = paths.llama_bin("llama-quantize")
    print(f"[export] {qbin} {f16.name} -> Q2_0 (embd+output also Q2_0 to match 2.03 GiB) ...", flush=True)
    subprocess.run([str(qbin),
                    "--output-tensor-type", "Q2_0",
                    "--token-embedding-type", "Q2_0",
                    str(f16), str(q2), "Q2_0"], check=True)
    print(f"[export] DONE: {q2} ({q2.stat().st_size/1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
