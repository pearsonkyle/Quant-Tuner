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

    # 1) overwrite trained-layer weights with the QAT latents, reporting the
    #    artifact-level code-flip stats: the shipped base is exactly ternary, so
    #    pre-overwrite codes are sign(W); the trained latents re-ternarize to
    #    the codes llama-quantize will store. Zero flips across the board means
    #    the run only drifted scales (the lr-too-low signature — see the audit).
    blob = torch.load(args.latents, map_location="cpu", weights_only=False)
    latents = blob["latents"]
    sd = dict(model.named_parameters())
    n_over = 0
    tot_flips = 0
    tot_codes = 0
    per_layer: dict[str, tuple[int, int, float]] = {}
    with torch.no_grad():
        for k, v in latents.items():
            base_key = k.replace(".linear.weight", ".weight")
            if base_key not in sd:
                continue
            if ".linear.weight" in k:  # ternary latent: measure flips + scale drift
                base_codes = torch.sign(sd[base_key].data)
                new_codes, new_scale, _ = ternarize_group(v.to(torch.float32))
                _, base_scale, _ = ternarize_group(sd[base_key].data)
                flips = int((new_codes != base_codes).sum())
                drift = float(((new_scale - base_scale).abs()
                               / base_scale.clamp_min(1e-8)).mean())
                per_layer[k] = (flips, base_codes.numel(), drift)
                tot_flips += flips
                tot_codes += base_codes.numel()
            sd[base_key].data.copy_(v.to(torch.float32))
            n_over += 1
    lf, ll = blob.get("loss_first"), blob.get("loss_last")
    loss_note = f" (loss {lf:.3f}->{ll:.3f})" if lf is not None and ll is not None else ""
    print(f"[export] overwrote {n_over}/{len(latents)} trained tensors{loss_note} "
          f"(step {blob.get('step','?')})", flush=True)
    if tot_codes:
        worst = sorted(per_layer.items(), key=lambda kv: -kv[1][0])[:5]
        for k, (fl, n, dr) in worst:
            print(f"[export]   {k}: {fl} flips ({100*fl/n:.4f}%) scale-drift {dr*100:.2f}%",
                  flush=True)
        print(f"[export] TOTAL code flips vs shipped: {tot_flips}/{tot_codes} "
              f"({100*tot_flips/max(1,tot_codes):.4f}%)", flush=True)

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

    # The unpacked tokenizer ships NO chat_template, so convert_hf_to_gguf bakes a
    # thinking-enabled Qwen3 default that breaks tool-use in the agent harness.
    # Restore the ORIGINAL prism Ternary template (extracted from the shipped GGUF)
    # so the QAT model behaves identically to the baseline in llama-server.
    tmpl = REPO / "out" / "exp-057" / "chat_template.jinja"
    if tmpl.exists():
        tok.chat_template = tmpl.read_text()
        print("[export] restored original prism chat template", flush=True)

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
