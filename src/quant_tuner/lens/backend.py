"""jlens-server lifecycle: spawn, health-check, tear down.

Mirrors :mod:`quant_tuner.eval.server` (which does the same for llama-server)
and reuses its ``free_port``/``wait_for_health``. The context manager yields a
connected :class:`~quant_tuner.lens.client.NativeClient` and asserts the
architecture self-check (``l_out_ok``) before anyone captures through it —
an unsupported arch fails loudly at startup, not with silent empty grids.

jlens-server is llama-server-compatible on ``/v1``, so the client's
``base_url`` can also be handed to :func:`quant_tuner.eval.toolcall.
run_toolcall_eval` to run the standard tool-call eval *with a live
intervention set active* (see ``NativeClient.live_interventions_set``).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from quant_tuner.eval.server import free_port, wait_for_health
from quant_tuner.lens.client import NativeClient
from quant_tuner.paths import jlens_server_bin

logger = logging.getLogger(__name__)

DEFAULT_CTX = 8192


def spawn_jlens_server(
    model_path: Path,
    *,
    port: int,
    ctx: int = DEFAULT_CTX,
    ngl: int = 99,
    chunk: int = 512,
    threads: int = 0,
    log_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start jlens-server on ``port`` and return the process (unwaited)."""
    cmd = [
        str(jlens_server_bin()),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx),
        "--chunk", str(chunk),
        "-ngl", str(ngl),
    ]
    if threads:
        cmd += ["-t", str(threads)]
    if extra_args:
        cmd += list(extra_args)
    logger.info("starting jlens-server: %s", " ".join(cmd))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # the handle is owned by the child process for its lifetime, so it must
        # outlive this function — a context manager would close it too early
        log_f = open(log_path, "w")  # noqa: SIM115
        return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextmanager
def running_jlens_server(
    model_path: Path,
    *,
    ctx: int = DEFAULT_CTX,
    ngl: int = 99,
    chunk: int = 512,
    threads: int = 0,
    log_path: Path | None = None,
    startup_timeout: float = 600.0,
    require_l_out: bool = True,
) -> Iterator[NativeClient]:
    """Spawn jlens-server for ``model_path``, yield a client, tear down.

    ``startup_timeout`` is generous by default: a 31B GGUF takes a while to
    mmap + warm up, and ``wait_for_health`` bails early anyway if the process
    dies (bad GGUF, port clash).
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = spawn_jlens_server(
        model_path, port=port, ctx=ctx, ngl=ngl, chunk=chunk,
        threads=threads, log_path=log_path,
    )
    try:
        wait_for_health(base_url, timeout=startup_timeout, proc=proc)
        client = NativeClient(base_url)
        props = client.props()
        if require_l_out and not props.get("l_out_ok", False):
            raise RuntimeError(
                f"jlens-server reports l_out_ok=false for {model_path} — this "
                "architecture does not expose per-layer residual (l_out) tensors; "
                "lens capture would return nothing."
            )
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
