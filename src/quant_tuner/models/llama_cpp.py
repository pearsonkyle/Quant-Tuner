"""Thin subprocess wrappers around llama.cpp binaries.

The build is expected at $LLAMA_CPP_DIR/build/bin or vendor/llama.cpp/build/bin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from quant_tuner.paths import llama_bin


def run(cmd: list[str | Path], log: Path | None = None, check: bool = True) -> str:
    """Execute a llama.cpp binary, tee stdout/stderr to `log` if given. Returns combined output."""
    args = [str(c) for c in cmd]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}):\n{out[-2000:]}")
    return out


def quantize(
    f16_gguf: Path,
    out_gguf: Path,
    quant_type: str,
    imatrix: Path | None = None,
    log: Path | None = None,
) -> Path:
    cmd: list[str | Path] = [llama_bin("llama-quantize")]
    if imatrix is not None:
        cmd += ["--imatrix", imatrix]
    cmd += [f16_gguf, out_gguf, quant_type]
    run(cmd, log=log)
    return out_gguf


def perplexity_kld(
    model: Path,
    dataset: Path,
    baseline: Path,
    ctx: int = 8192,
    n_tokens: int | None = None,
    log: Path | None = None,
) -> str:
    """Run llama-perplexity in KLD mode against a saved baseline.

    NOTE: unlike llama-imatrix, llama-perplexity has NO ``--parse-special`` —
    chat-control markers in ``dataset`` (``<|im_start|>`` etc.) tokenize as
    plain BPE text, a distribution the model never sees at inference. Prefer
    raw-text eval corpora (``scripts/build_corpora.py``'s ``corpus.eval.txt``,
    wired via ``bench.eval_corpus``) over chat-templated ones. Comparisons
    between quants on the same file remain internally consistent either way.
    """
    cmd: list[str | Path] = [
        llama_bin("llama-perplexity"),
        "-m", model,
        "-f", dataset,
        "-c", str(ctx),
        "--kl-divergence",
        "--kl-divergence-base", baseline,
    ]
    if n_tokens is not None:
        cmd += ["-n", str(n_tokens)]
    return run(cmd, log=log)


def perplexity_baseline(
    model: Path,
    dataset: Path,
    baseline_out: Path,
    ctx: int = 8192,
    n_tokens: int | None = None,
    log: Path | None = None,
) -> Path:
    cmd: list[str | Path] = [
        llama_bin("llama-perplexity"),
        "-m", model,
        "-f", dataset,
        "-c", str(ctx),
        "--kl-divergence-base", baseline_out,
    ]
    if n_tokens is not None:
        cmd += ["-n", str(n_tokens)]
    run(cmd, log=log)
    return baseline_out


def imatrix(
    model: Path,
    calibration_file: Path,
    out: Path,
    ctx: int = 512,
    log: Path | None = None,
    extra_args: list[str] | None = None,
    parse_special: bool = True,
) -> Path:
    cmd: list[str | Path] = [
        llama_bin("llama-imatrix"),
        "-m", model,
        "-f", calibration_file,
        "-o", out,
        "-c", str(ctx),
    ]
    # Our calibration corpus is chat-templated (split.write_corpus emits
    # sessions rendered through apply_chat_template, full of <|im_start|>
    # etc). Without --parse-special, llama-imatrix tokenizes those control
    # markers as literal text (e.g. <|im_start|> -> 6 BPE pieces) instead of
    # the single special-token IDs the model sees at inference, so the
    # collected activation stats reflect a token distribution that never
    # occurs in production. Default it on; harmless on raw-text corpora
    # (wiki) since they contain no special tokens to parse.
    if parse_special:
        cmd.append("--parse-special")
    if extra_args:
        cmd += extra_args
    run(cmd, log=log)
    return out


def bench(
    model: Path,
    n_prompt: int = 2048,
    n_gen: int = 128,
    repetitions: int = 5,
    log: Path | None = None,
) -> str:
    cmd: list[str | Path] = [
        llama_bin("llama-bench"),
        "-m", model,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-o", "json",
    ]
    return run(cmd, log=log)
