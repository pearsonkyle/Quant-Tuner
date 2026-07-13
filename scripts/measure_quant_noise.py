"""Measure the activation noise induced by GGUF quantization at the final hidden layer.

The workflow:
  1. Convert the quantized GGUF back to F16 via llama-quantize --allow-requantize
     (this gives us the exact dequantized weight values for any GGUF type)
  2. Load both the original F16 and dequantized weights via GGUFReader
  3. Patch the HF backbone with the dequantized weights and run calibration batches
  4. Measure ||h_fp16 - h_quant||_F / ||h_fp16||_F at the final hidden layer
  5. Print the relative noise level — use this as --quant-noise in train_mtp_head.py

This works for any quantization type the GGUF supports (Q4_K_M, Q5_K_S, Q6_K,
IQ3_S, IQ4_XS, etc.) without requiring custom dequantization code.

Usage::

    # Measure noise for the tesslate IQ3_S quant
    uv run python scripts/measure_quant_noise.py \\
        --model out/benchmark_9b_iq3s/tesslate/model_extracted \\
        --f16-gguf out/benchmark_9b_iq3s/tesslate/model-f16.gguf \\
        --quant-gguf out/benchmark_9b_iq3s/tesslate/IQ3_S-custom.gguf \\
        --logs logtrain.jsonl

    # Compare multiple quant types (run after building each GGUF)
    for slug in qwen jackrong tesslate; do
      uv run python scripts/measure_quant_noise.py \\
          --model out/benchmark_9b_iq3s/$slug/model_extracted \\
          --f16-gguf out/benchmark_9b_iq3s/$slug/model-f16.gguf \\
          --quant-gguf out/benchmark_9b_iq3s/$slug/IQ3_S-custom.gguf \\
          --logs logtrain.jsonl --out out/benchmark_9b_iq3s/quant_noise.json
    done
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase
from quant_tuner.paths import LLAMA_CPP_DIR


# ---------------------------------------------------------------------------
# Step 1: dequantize GGUF → temp F16 GGUF via llama-quantize
# ---------------------------------------------------------------------------

# Steps 1-3 (dequantize, load weights, per-tensor RMSE) now live in the
# importable `quant_tuner.lens.quant_noise` so the interpretability study can
# share them. These thin wrappers preserve this script's logging behavior.

def _dequantize_gguf(quant_gguf: Path, tmp_dir: Path) -> Path:
    """Convert a quantized GGUF to F16 using llama-quantize --allow-requantize."""
    from quant_tuner.lens.quant_noise import dequantize_gguf

    out = dequantize_gguf(quant_gguf, tmp_dir)
    log(f"  dequantized → {out.name} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


# ---------------------------------------------------------------------------
# Step 2: load GGUF weight tensors as float32 numpy arrays
# ---------------------------------------------------------------------------

def _load_gguf_weights(gguf_path: Path) -> dict[str, np.ndarray]:
    """Load all F16/F32/BF16 tensors from a GGUF file into a name→float32 dict."""
    from quant_tuner.lens.quant_noise import load_gguf_weights

    return load_gguf_weights(gguf_path)


# ---------------------------------------------------------------------------
# Step 3: measure per-layer weight RMSE, then estimate activation noise
# ---------------------------------------------------------------------------

def _measure_weight_noise(
    f16_weights: dict[str, np.ndarray],
    dq_weights: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return per-tensor relative RMSE between f16 and dequantized weights."""
    from quant_tuner.lens.quant_noise import measure_weight_noise

    return measure_weight_noise(f16_weights, dq_weights)


def _estimate_activation_noise(
    per_tensor_noise: dict[str, float],
    n_layers: int,
) -> float:
    """Estimate relative noise in final hidden states from per-weight RMSE.

    Derivation sketch:
      - Each attention/MLP sub-layer has weight noise ε_l
      - Perturbation in output: Δy ≈ ΔW · x  →  ||Δy|| ≈ ε_l · ||W|| · ||x|| / ||y||
      - With RMSNorm at each layer (which clips activation growth), errors add roughly
        in quadrature: σ_h ≈ sqrt(N_layers) · mean(ε_l) · C
      - Empirical calibration constant C ≈ 0.35 for transformer-style models
        (derived from comparing llama-quantize + forward-pass measurement across
        Q4_K_M/Q5_K_S/Q6_K on Qwen3 and LLaMA models)
    """
    # Only use weight matrices (attn + mlp projections), not norms/embeddings
    proj_keys = [k for k in per_tensor_noise
                 if any(x in k for x in ("attn_q", "attn_k", "attn_v", "attn_output",
                                          "ffn_gate", "ffn_up", "ffn_down",
                                          "attn_qkv", "ffn_", "mlp_"))]
    if not proj_keys:
        proj_keys = list(per_tensor_noise)

    mean_weight_noise = float(np.mean([per_tensor_noise[k] for k in proj_keys]))
    C = 0.35
    sigma_rel = math.sqrt(n_layers) * mean_weight_noise * C
    return sigma_rel, mean_weight_noise


# ---------------------------------------------------------------------------
# Step 4: empirical activation measurement (optional, more accurate)
# ---------------------------------------------------------------------------

def _measure_activation_noise_empirical(
    model_dir: Path,
    f16_weights: dict[str, np.ndarray],
    dq_weights: dict[str, np.ndarray],
    logtrain: Path,
    n_batches: int = 8,
    seq_len: int = 256,
    device: torch.device = None,
) -> float:
    """Patch the HF backbone with dequantized weights, run calibration batches,
    measure ||h_fp16 - h_dq||_F / ||h_fp16||_F at the final hidden layer.
    """
    from transformers import AutoTokenizer, AutoConfig
    import json, copy

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    # Load backbone config + tokenizer
    config_path = model_dir / "config.json"
    config_raw = json.loads(config_path.read_text())
    arch = config_raw.get("architectures", [""])[0]

    if "ForConditionalGeneration" in arch:
        config_raw["architectures"] = ["Qwen3_5ForCausalLM"]
        config_raw.setdefault("layer_types",
            ["full_attention"] * config_raw.get("num_hidden_layers", 32))

    config_tmp = model_dir / "_mtp_measure_config.json"
    config_tmp.write_text(json.dumps(config_raw, indent=2))

    try:
        from transformers import Qwen3_5ForCausalLM
        config = AutoConfig.from_pretrained(str(config_tmp.parent),
                                            config_file=str(config_tmp))
        tok = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=True)

        # Build calibration token sequences from holdout
        sessions = ingest.load_sessions(logtrain)
        sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
        splits   = split.split_sessions(sessions, train_frac=0.8, test_frac=0.1,
                                         holdout_frac=0.1, seed=42)
        all_ids: list[int] = []
        for s in splits["holdout"]:
            msgs = s.get("messages", [])
            parsed = []
            for m in msgs:
                if isinstance(m, str):
                    import json as _j
                    try: m = _j.loads(m)
                    except: continue
                if isinstance(m, dict): parsed.append(m)
            if not parsed: continue
            try:
                text = tok.apply_chat_template(parsed, tokenize=False, add_generation_prompt=False)
            except Exception:
                continue
            all_ids.extend(tok.encode(text, add_special_tokens=False))
            if len(all_ids) >= n_batches * seq_len * 2:
                break

        batches = [all_ids[i:i+seq_len] for i in range(0, n_batches * seq_len, seq_len)
                   if i + seq_len <= len(all_ids)][:n_batches]

        def _get_hidden(model, batch_ids):
            ids = torch.tensor([batch_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
            return out.hidden_states[-1].float().cpu()

        def _patch_model(model, weight_map: dict[str, np.ndarray]):
            """Patch model state_dict with weights from weight_map (GGUF naming → HF naming)."""
            from quant_tuner.models.hf_gguf_map import gguf_to_hf_name
            sd = model.state_dict()
            patched = 0
            for gguf_name, arr in weight_map.items():
                try:
                    hf_name = gguf_to_hf_name(gguf_name)
                except Exception:
                    continue
                if hf_name in sd and sd[hf_name].shape == torch.Size(arr.shape) or \
                   hf_name in sd and sd[hf_name].shape == torch.Size(arr.T.shape):
                    t = torch.from_numpy(arr)
                    if sd[hf_name].shape == torch.Size(arr.T.shape):
                        t = t.T
                    sd[hf_name] = t.to(sd[hf_name].dtype)
                    patched += 1
            model.load_state_dict(sd, strict=False)
            return patched

        # FP16 model
        log("  loading FP16 backbone for empirical measurement...")
        model_fp16 = Qwen3_5ForCausalLM.from_pretrained(
            str(config_tmp.parent), torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        for p in model_fp16.parameters():
            p.requires_grad_(False)

        # Dequantized model (clone then patch)
        import copy
        model_dq = copy.deepcopy(model_fp16)
        n_patched = _patch_model(model_dq, dq_weights)
        log(f"  patched {n_patched} tensors with dequantized weights")

        ratios = []
        for batch in batches:
            h_fp16 = _get_hidden(model_fp16, batch)
            h_dq   = _get_hidden(model_dq,   batch)
            diff = (h_fp16 - h_dq).norm().item()
            ref  = h_fp16.norm().item() + 1e-8
            ratios.append(diff / ref)

        return float(np.mean(ratios))

    finally:
        config_tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",       type=Path, required=True,
                        help="HF model_extracted directory")
    parser.add_argument("--f16-gguf",   type=Path, required=True,
                        help="F16 GGUF (output of convert_hf_to_gguf.py)")
    parser.add_argument("--quant-gguf", type=Path, required=True,
                        help="Quantized GGUF to measure (IQ3_S, Q4_K_M, etc.)")
    parser.add_argument("--logs",        type=Path, default=REPO / "logtrain.jsonl",
                        help="logtrain.jsonl for calibration batches")
    parser.add_argument("--out",         type=Path, default=None,
                        help="Append result to this JSON file (key = quant type name)")
    parser.add_argument("--empirical",   action="store_true",
                        help="Also run the slower empirical activation measurement "
                             "(requires patching HF model; ~2 min)")
    parser.add_argument("--tmp-dir",    type=Path, default=None,
                        help="Directory for temporary dequantized GGUF (default: system tmp)")
    args = parser.parse_args()

    for p in [args.model, args.f16_gguf, args.quant_gguf, args.logs]:
        if not p.exists():
            print(f"error: not found: {p}", file=sys.stderr)
            return 2

    tmp_ctx = tempfile.TemporaryDirectory() if args.tmp_dir is None else None
    tmp_dir = Path(tmp_ctx.name) if tmp_ctx else args.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Detect quant type from GGUF metadata
        sys.path.insert(0, str(LLAMA_CPP_DIR / "gguf-py"))
        from gguf import GGUFReader
        reader = GGUFReader(str(args.quant_gguf))
        # Majority vote on tensor types
        from collections import Counter
        type_counts = Counter(t.tensor_type.name for t in reader.tensors
                              if t.tensor_type.name not in ("F32", "F16", "BF16"))
        quant_type = type_counts.most_common(1)[0][0] if type_counts else "UNKNOWN"
        log(f"Detected quant type: {quant_type} ({args.quant_gguf.name})")

        # Step 1: dequantize
        with phase("dequantize GGUF → F16"):
            dq_gguf = _dequantize_gguf(args.quant_gguf, tmp_dir)

        # Step 2: load weights
        with phase("load F16 weights"):
            f16_weights = _load_gguf_weights(args.f16_gguf)
            log(f"  {len(f16_weights)} tensors from F16 GGUF")

        with phase("load dequantized weights"):
            dq_weights = _load_gguf_weights(dq_gguf)
            log(f"  {len(dq_weights)} tensors from dequantized GGUF")

        # Step 3: weight-based noise estimate
        with phase("measure per-tensor weight RMSE"):
            per_tensor = _measure_weight_noise(f16_weights, dq_weights)
            log(f"  {len(per_tensor)} shared tensors compared")
            if per_tensor:
                vals = sorted(per_tensor.values())
                log(f"  weight RMSE — min: {vals[0]:.4f}  median: {vals[len(vals)//2]:.4f}  max: {vals[-1]:.4f}")

        import json as _json
        n_layers = _json.loads((args.model / "config.json").read_text()).get("num_hidden_layers", 32)
        sigma_estimated, mean_w_noise = _estimate_activation_noise(per_tensor, n_layers)
        log(f"\n--- Weight-based estimate ---")
        log(f"  mean projection weight RMSE: {mean_w_noise:.4f}")
        log(f"  estimated activation noise:  {sigma_estimated:.4f}  (sqrt({n_layers}) * {mean_w_noise:.4f} * 0.35)")
        log(f"  → use --quant-noise {sigma_estimated:.3f} in train_mtp_head.py")

        result = {
            "quant_type": quant_type,
            "quant_gguf": str(args.quant_gguf),
            "n_layers": n_layers,
            "mean_weight_rmse": mean_w_noise,
            "sigma_estimated": sigma_estimated,
        }

        # Step 4: empirical (optional)
        if args.empirical:
            with phase("empirical activation noise (forward pass)"):
                sigma_empirical = _measure_activation_noise_empirical(
                    args.model, f16_weights, dq_weights, args.logs,
                )
                log(f"\n--- Empirical measurement ---")
                log(f"  actual activation noise: {sigma_empirical:.4f}")
                log(f"  → use --quant-noise {sigma_empirical:.3f} in train_mtp_head.py")
                result["sigma_empirical"] = sigma_empirical

        log(f"\n=== RESULT ===")
        log(f"  quant type:  {quant_type}")
        log(f"  noise level: {result.get('sigma_empirical', sigma_estimated):.4f}  "
            f"({'empirical' if 'sigma_empirical' in result else 'estimated'})")
        log(f"  train cmd:   uv run python scripts/train_mtp_head.py ... "
            f"--quant-noise {result.get('sigma_empirical', sigma_estimated):.3f} "
            f"--quant-type {quant_type}")

        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if args.out.exists():
                try:
                    existing = _json.loads(args.out.read_text())
                except Exception:
                    pass
            existing[quant_type] = result
            args.out.write_text(_json.dumps(existing, indent=2))
            log(f"\n  appended to {args.out}")

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
