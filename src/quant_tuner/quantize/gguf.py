"""GGUF quantization via llama-quantize.

The quant `type` is passed through to llama-quantize unchanged — to switch from
Q4_K_M to IQ4_XS, Q5_K_M, IQ3_S, etc., change one string.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from quant_tuner.models import llama_cpp
from quant_tuner.paths import LLAMA_CPP_DIR


def quantize(
    f16_gguf: Path,
    out_gguf: Path,
    quant_type: str,
    imatrix: Path | None = None,
    log: Path | None = None,
    tensor_types: dict[str, str] | None = None,
) -> Path:
    """Quantize an F16 GGUF to `quant_type`, optionally guided by an imatrix.

    ``tensor_types`` pins matching tensors to a different ggml type (by name
    substring) — e.g. ``{"nextn": "q8_0"}`` keeps the MTP draft head
    near-lossless while the trunk goes to a 2-bit type.
    """
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    return llama_cpp.quantize(
        f16_gguf, out_gguf, quant_type,
        imatrix=imatrix, log=log, tensor_types=tensor_types,
    )


def template_stamp(
    gguf_path: Path, template: Path | None, general_name: str | None = None,
) -> Path:
    """Path of the marker recording which metadata is baked into ``gguf_path``.

    The inputs' SHA-256 is in the **filename** on purpose: `step()` is
    existence-based, so a stamp named after the stage alone would make a
    changed template a silent no-op on re-run — exactly the stale-artifact
    trap the quant-GGUF naming convention exists to avoid.
    """
    h = hashlib.sha256()
    if template is not None:
        h.update(template.read_bytes())
    if general_name is not None:
        h.update(b"\0name=" + general_name.encode())
    return gguf_path.with_suffix(
        gguf_path.suffix + f".chat-template-{h.hexdigest()[:12]}.stamp"
    )


def set_metadata(
    gguf_path: Path,
    template: Path | None = None,
    general_name: str | None = None,
    log: Path | None = None,
) -> Path:
    """Rewrite ``gguf_path``'s embedded metadata in place.

    Runs llama.cpp's `gguf_new_metadata.py`, which copies the container with the
    named keys replaced — **no re-quantization**, so the weights are bit-identical
    (verified 2026-08-15 on the IQ2_M rung: 866/866 tensors unchanged in shape and
    dtype, exactly one KV *value* different; six other keys shifted only in
    `index`/`offset` because the template grew).

    Both keys go in **one** pass because the script rewrites the whole file —
    doing them separately means copying 10 GB twice.

    The copy goes to a sibling temp file and is then `os.replace`-d over the
    original, which is atomic on the same filesystem: an interrupted run leaves
    either the old GGUF or the new one, never a truncated one.

    Writes the marker from `template_stamp` on success.
    """
    script = LLAMA_CPP_DIR / "gguf-py" / "gguf" / "scripts" / "gguf_new_metadata.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing gguf_new_metadata.py: {script}")
    if template is None and general_name is None:
        raise ValueError("set_metadata called with nothing to set")
    if template is not None and not template.exists():
        raise FileNotFoundError(f"Missing chat template: {template}")

    tmp = gguf_path.with_suffix(gguf_path.suffix + ".newtmpl.tmp")
    cmd = [sys.executable, str(script)]
    if template is not None:
        cmd += ["--chat-template-file", str(template)]
    if general_name is not None:
        # gguf_set_metadata.py CANNOT do this after the fact — its
        # gguf_scalar_to_np map has no STRING entry, so it refuses string
        # fields. A full rewrite is the only retrofit, which is why it rides
        # along here rather than becoming its own release-time pass.
        cmd += ["--general-name", general_name]
    cmd += [
        "--force",  # non-interactive: the whole point is that we already A/B'd it
        str(gguf_path), str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out)
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"gguf_new_metadata failed:\n{out[-2000:]}")
    os.replace(tmp, gguf_path)

    stamp = template_stamp(gguf_path, template, general_name)
    stamp.write_text(f"chat_template={template}\ngeneral_name={general_name}\n")
    return gguf_path
