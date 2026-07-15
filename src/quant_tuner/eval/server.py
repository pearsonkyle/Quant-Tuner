"""llama-server lifecycle helpers for OpenAI-compatible tool-call evaluation."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from quant_tuner.paths import llama_bin


def free_port() -> int:
    """Pick an unused TCP port the OS will assign on bind."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(
    base_url: str,
    timeout: float = 120.0,
    proc: subprocess.Popen | None = None,
) -> None:
    """Block until llama-server's ``/health`` returns 200, or raise ``TimeoutError``.

    When ``proc`` is given, a server that exits before becoming healthy (bad
    GGUF, port clash, OOM) raises immediately instead of burning the timeout.
    """
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(f"llama-server at {url} did not become healthy in {timeout}s ({last_err})")


def spawn_server(
    model_path: Path,
    *,
    port: int,
    ctx: int = 8192,
    ngl: int = 99,
    log_path: Path | None = None,
    chat_template_kwargs: str | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int | None = None,
    mmproj_path: Path | None = None,
) -> subprocess.Popen:
    """Start ``llama-server`` for ``model_path`` and return the process handle.

    The caller is responsible for terminating the process and for calling
    :func:`wait_for_health` before issuing requests. Use :func:`running_server`
    for an auto-managed context.

    ``chat_template_kwargs`` is forwarded to llama-server's
    ``--chat-template-kwargs`` flag (JSON string, e.g.
    ``'{"enable_thinking":false}'`` to disable reasoning on Qwen3-family
    templates).

    ``spec_type`` / ``spec_draft_n_max`` map to llama-server's ``--spec-type``
    and ``--spec-draft-n-max`` flags for speculative decoding (e.g. MTP draft
    heads bundled into the GGUF: ``spec_type="draft-mtp"``).

    ``mmproj_path`` points to a multimodal projector GGUF (``--mmproj``).
    Required for vision models (e.g. LLaVA, Gemma-3, Qwen2-VL).
    """
    binary = llama_bin("llama-server")  # raises if not built
    cmd = [
        str(binary),
        "-m", str(model_path),
        "--jinja",
        "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(ngl),
        "-fa", "on",            # force flash attention (default 'auto') — faster + less KV memory
        "-ub", "1024",          # larger physical batch speeds prompt processing on long agent contexts
        # Server-side repetition penalty: low-bit (2-bit) quants loop in agent
        # loops, generating toward the token cap (minutes per step). The
        # openai-agents backend can't forward a penalty, so set it here as the
        # default; callers that send their own (tool-call reps send 1.0) override.
        # Tunable via env for models that loop harder (e.g. the ternary QAT model,
        # which repeats whole commands): QT_REPEAT_PENALTY / QT_REPEAT_LAST_N /
        # QT_PRESENCE_PENALTY / QT_FREQUENCY_PENALTY (long-range loop control).
        "--repeat-penalty", os.environ.get("QT_REPEAT_PENALTY", "1.1"),
        "--repeat-last-n", os.environ.get("QT_REPEAT_LAST_N", "256"),
        "--host", "127.0.0.1",
    ]
    if os.environ.get("QT_PRESENCE_PENALTY"):
        cmd += ["--presence-penalty", os.environ["QT_PRESENCE_PENALTY"]]
    if os.environ.get("QT_FREQUENCY_PENALTY"):
        cmd += ["--frequency-penalty", os.environ["QT_FREQUENCY_PENALTY"]]
    if mmproj_path is not None:
        cmd += ["--mmproj", str(mmproj_path)]
    if chat_template_kwargs is not None:
        cmd += ["--chat-template-kwargs", chat_template_kwargs]
        # `enable_thinking` via --chat-template-kwargs is DEPRECATED and silently
        # ignored by recent templates (e.g. gemma-4 / peg-gemma4 keeps thinking
        # ON). The authoritative switch is --reasoning; translate so every caller
        # that already passes {"enable_thinking":false} actually disables it.
        _low = chat_template_kwargs.replace(" ", "").lower()
        if '"enable_thinking":false' in _low:
            cmd += ["--reasoning", "off"]
        elif '"enable_thinking":true' in _low:
            cmd += ["--reasoning", "on"]
    if spec_type is not None:
        cmd += ["--spec-type", spec_type]
    if spec_draft_n_max is not None:
        cmd += ["--spec-draft-n-max", str(spec_draft_n_max)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_fh:  # child keeps its own dup'd fd
            return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return subprocess.Popen(cmd)


@contextmanager
def running_server(
    model_path: Path,
    *,
    ctx: int = 8192,
    ngl: int = 99,
    log_path: Path | None = None,
    startup_timeout: float = 120.0,
    chat_template_kwargs: str | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int | None = None,
    mmproj_path: Path | None = None,
):
    """Context manager: spawn llama-server, wait for health, yield ``base_url``, then terminate.

    Pass ``chat_template_kwargs='{"enable_thinking":false}'`` to disable
    reasoning on Qwen3-family templates during benchmark eval.

    Pass ``mmproj_path`` to enable multimodal (vision) input for models that
    require a separate projector GGUF (``--mmproj`` flag).
    """
    port = free_port()
    proc = spawn_server(
        model_path, port=port, ctx=ctx, ngl=ngl,
        log_path=log_path, chat_template_kwargs=chat_template_kwargs,
        spec_type=spec_type, spec_draft_n_max=spec_draft_n_max,
        mmproj_path=mmproj_path,
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        wait_for_health(base_url, timeout=startup_timeout, proc=proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
