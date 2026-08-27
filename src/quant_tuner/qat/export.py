"""Export trained ternary latents -> Q2_0 GGUF (runnable on the prism fork).

Loads the unpacked base model, overwrites the trained layers' weights with the QAT latents
(reporting artifact-level code-flip stats vs the shipped weights — zero flips means the run
only drifted scales, the lr-too-low signature), then ternarizes EVERY qualifying linear with
the SAME TWN quantizer the forward used (frozen layers = bit-exact no-op; trained layers =
their final ternary codes). The result is a pure-ternary HF model -> F16 GGUF -> Q2_0
(lossless, weights already on the grid). Q2_0 output needs the prism llama-quantize (type 41).

The unpacked tokenizer ships NO chat_template, so the F16 conversion would bake a
thinking-enabled Qwen3 default that breaks tool-use; the original prism template is restored
from ``out/exp-057/chat_template.jinja`` so the QAT model behaves like the baseline.

The CLI shim is ``scripts/exp057_qat_export.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from quant_tuner.qat.ternary import ternarize_group
from quant_tuner.quantize import convert

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "out" / "exp-057" / "model"
EXP057 = REPO / "out" / "exp-057"


def export_qat(latents: Path, tag: str = "qat", *, model_dir: Path = MODEL,
               exp_dir: Path = EXP057) -> Path:
    """Bake ``latents`` into the base model and requantize to Q2_0. Returns the GGUF path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_hf = exp_dir / f"model_{tag}"
    f16 = exp_dir / f"Ternary-Bonsai-8B-{tag}-F16.gguf"
    q2 = exp_dir / f"Ternary-Bonsai-8B-{tag}-Q2_0.gguf"

    print(f"[export] loading base {model_dir} (fp32, cpu)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)

    # 1) overwrite trained-layer weights with the QAT latents + report code flips
    blob = torch.load(latents, map_location="cpu", weights_only=False)
    latent_sd = blob["latents"]
    sd = dict(model.named_parameters())
    n_over = tot_flips = tot_codes = 0
    per_layer: dict[str, tuple[int, int, float]] = {}
    with torch.no_grad():
        for k, v in latent_sd.items():
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
    print(f"[export] overwrote {n_over}/{len(latent_sd)} trained tensors{loss_note} "
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

    # Restore the original prism chat template. This is NOT optional and must not be a
    # silent skip: the unpacked tokenizer ships no chat_template, so the conversion bakes
    # a thinking-enabled Qwen3 default instead. The resulting GGUF loads, serves, and
    # answers — it just emits <think> and takes zero tool steps, which reads as a model
    # that lost its agentic ability rather than as a missing file. Fail here instead.
    tmpl = exp_dir / "chat_template.jinja"
    if not tmpl.exists():
        raise FileNotFoundError(
            f"{tmpl} is missing. Without it the F16 conversion bakes a thinking-enabled "
            f"Qwen3 default and the exported model emits <think> and never tool-calls — "
            f"a silent, benchmark-shaped failure. Copy it from the unpacked model dir: "
            f"cp {model_dir / 'chat_template.jinja'} {tmpl}")
    tok.chat_template = tmpl.read_text()
    print("[export] restored original prism chat template", flush=True)

    out_hf.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_hf)
    tok.save_pretrained(out_hf)
    print(f"[export] saved HF -> {out_hf}", flush=True)

    # 3) F16 GGUF, then Q2_0 (prism llama-quantize)
    print("[export] convert -> F16 GGUF ...", flush=True)
    convert.hf_to_f16_gguf(out_hf, f16, log=exp_dir / f"convert_{tag}.log")

    from quant_tuner import paths
    qbin = paths.llama_bin("llama-quantize")
    print(f"[export] {qbin} {f16.name} -> Q2_0 (embd+output also Q2_0) ...", flush=True)
    subprocess.run([str(qbin),
                    "--output-tensor-type", "Q2_0",
                    "--token-embedding-type", "Q2_0",
                    str(f16), str(q2), "Q2_0"], check=True)
    print(f"[export] DONE: {q2} ({q2.stat().st_size/1024**3:.2f} GiB)")
    return q2
