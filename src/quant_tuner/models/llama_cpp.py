"""Thin subprocess wrappers around llama.cpp binaries.

The build is expected at $LLAMA_CPP_DIR/build/bin or vendor/llama.cpp/build/bin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from quant_tuner.paths import ensure_gguf_py, llama_bin


def gguf_n_vocab(gguf_path: Path) -> int | None:
    """Read the vocabulary size out of a GGUF, or ``None`` if it can't be determined.

    Used to predict the ``llama-imatrix`` perplexity-path integer overflow (see
    :func:`imatrix`). Reads only header/metadata — the GGUF is memory-mapped, so
    this does not touch the tensor payload.
    """
    try:
        # The vendored gguf-py is not installed into the venv — without this the
        # import fails, we cannot predict the overflow, and the run segfaults two
        # hours into imatrix collection.
        ensure_gguf_py()
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        toks = reader.fields.get("tokenizer.ggml.tokens")
        if toks is not None and toks.data is not None:
            return len(toks.data)
        for name, field in reader.fields.items():
            if name.endswith(".vocab_size") and field.parts:
                return int(field.parts[field.data[0]][0])
    except Exception:  # noqa: BLE001 - best-effort probe, never fatal
        return None
    return None


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
    tensor_types: dict[str, str] | None = None,
) -> Path:
    cmd: list[str | Path] = [llama_bin("llama-quantize")]
    if imatrix is not None:
        cmd += ["--imatrix", imatrix]
    # Per-tensor type overrides (e.g. keep MTP/nextn near-lossless at q8_0 while
    # the trunk goes to a 2-bit type). llama-quantize matches by tensor-name
    # substring: ``--tensor-type nextn=q8_0``. Options precede the positionals.
    for name, ttype in (tensor_types or {}).items():
        cmd += ["--tensor-type", f"{name}={ttype}"]
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
    process_output: bool = True,
    no_ppl: bool | None = None,
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
    # llama-imatrix defaults process_output=false (imatrix.cpp only collects tensors whose
    # name starts with "blk."), so output.weight — the single largest quantized tensor, and
    # the one that directly shapes the token distribution KLD measures — is quantized with
    # NO activation statistics. llama-quantize says so on stderr ("did not find weights for
    # output.weight") and then proceeds, so it is silent unless you read the log. Collect
    # it. (token_embd.weight is a lookup, not a matmul, so it is never collectable.)
    # Defaulting this ON changes calibration vs. every run built before it — pass
    # process_output=False to reproduce the pre-exp-060 published numbers byte-for-byte.
    if process_output:
        cmd.append("--process-output")
    # llama-imatrix's perplexity readout indexes an [n_ctx, n_vocab] logits buffer
    # as `all_logits + first * n_vocab` with `first = n_ctx/2` — both plain `int`
    # (tools/imatrix/imatrix.cpp:911, llama.cpp pin f3e182816). On a large-vocab
    # model at high ctx that product overflows INT_MAX, the pointer offset goes
    # negative, and the process SIGSEGVs *after the first pass* — which looks like
    # an OOM or an unsupported architecture but is neither. Measured on
    # Qwen3.8-27B (n_vocab 248,320): ctx 8192/16384 fine, 24576/32768 die.
    # `--no-ppl` skips only that bookkeeping; the forward pass, and therefore every
    # activation statistic we actually want, is unchanged (and ~15% faster).
    # Real PPL/KLD comes from the bench stage against the eval corpora regardless.
    if no_ppl is None:
        n_vocab = gguf_n_vocab(model)
        if n_vocab is None:
            print(
                f"[imatrix] WARNING: could not read n_vocab from {model}; cannot check "
                f"the perplexity-path overflow. If this segfaults after the first pass, "
                f"pass no_ppl=True.",
                file=sys.stderr,
            )
            no_ppl = False
        else:
            no_ppl = (ctx // 2) * n_vocab >= 2**31
            if no_ppl:
                print(
                    f"[imatrix] n_ctx/2 * n_vocab = {ctx // 2} * {n_vocab} exceeds INT_MAX "
                    f"— passing --no-ppl to avoid the perplexity-path overflow segfault.",
                    file=sys.stderr,
                )
    if no_ppl:
        cmd.append("--no-ppl")
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
