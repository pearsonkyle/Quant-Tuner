"""llama-server lifecycle helpers for OpenAI-compatible tool-call evaluation."""

from __future__ import annotations

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


def wait_for_health(base_url: str, timeout: float = 120.0) -> None:
    """Block until llama-server's ``/health`` returns 200, or raise ``TimeoutError``."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
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
) -> subprocess.Popen:
    """Start ``llama-server`` for ``model_path`` and return the process handle.

    The caller is responsible for terminating the process and for calling
    :func:`wait_for_health` before issuing requests. Use :func:`running_server`
    for an auto-managed context.
    """
    binary = llama_bin("llama-server")  # raises if not built
    cmd = [
        str(binary),
        "-m", str(model_path),
        "--jinja",
        "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(ngl),
        "--host", "127.0.0.1",
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w")
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
):
    """Context manager: spawn llama-server, wait for health, yield ``base_url``, then terminate."""
    port = free_port()
    proc = spawn_server(model_path, port=port, ctx=ctx, ngl=ngl, log_path=log_path)
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        wait_for_health(base_url, timeout=startup_timeout)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
