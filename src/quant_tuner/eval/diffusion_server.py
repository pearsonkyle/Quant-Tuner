"""DiffusionGemma generation backend: an OpenAI-compatible shim over llama.cpp's
persistent ``llama-diffusion-gemma-server``.

``llama-server`` cannot run block-diffusion decoding, so the tool-call and
MMLU-Pro evals can't point at it for DiffusionGemma quants. This module keeps
their ``base_url`` contract intact instead of changing the evals:

  * :class:`LogitsServer` drives llama.cpp's ``llama-diffusion-gemma-server``
    (loads the GGUF **once**, then serves raw canvas logits for
    ``[prompt | canvas]`` requests over a stdin/file protocol — the PR-provided
    building block for Python diffusion drivers).
  * :func:`entropy_bound_generate` is a faithful numpy port of llama.cpp's
    ``diffusion_generate_entropy_bound`` (the DiffusionGemma reference
    sampler): per-step argmax/entropy/multinomial over the canvas,
    lowest-entropy acceptance within the entropy bound, renoise of the rest,
    self-conditioning between steps (cached server-side), block-autoregressive
    commit with EOG/repetition trimming. Deterministic per seed (not
    bit-identical to the C++ RNG stream).
  * :func:`running_diffusion_server` spawns a local HTTP shim implementing
    ``/health`` + ``/v1/chat/completions``: chat-template rendering (with
    tools) via the HF tokenizer, generation via the sampler, and Gemma-4
    tool-call markup (``<|tool_call>call:NAME{...}<tool_call|>``) parsed back
    into OpenAI ``tool_calls``.

Sampling: the entropy-bound schedule (t_max → t_min from GGUF metadata)
governs temperature; per-request ``seed`` is honored, other OpenAI sampling
knobs are accepted and ignored.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from quant_tuner.eval.server import free_port
from quant_tuner.paths import LLAMA_CPP_DIR, llama_bin

# --- entropy-bound parameters (GGUF metadata, CLI reference defaults) ------ #

@dataclass
class EbParams:
    canvas_length: int = 256
    max_steps: int = 48
    t_min: float = 0.4
    t_max: float = 0.8
    entropy_bound: float = 0.1
    stability_threshold: int = 1
    confidence_threshold: float = 0.005


def read_eb_params(gguf_path: Path) -> EbParams:
    """Entropy-bound sampler config from GGUF ``diffusion.*`` metadata."""
    p = str(LLAMA_CPP_DIR / "gguf-py")
    if p not in sys.path:
        sys.path.insert(0, p)
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))

    def field(key: str, default):
        f = reader.fields.get(key)
        if f is None:
            return default
        return type(default)(f.parts[f.data[0]][0])

    return EbParams(
        canvas_length=field("diffusion.canvas_length", 256),
        max_steps=field("diffusion.eb_max_steps", 48),
        t_min=field("diffusion.eb_t_min", 0.4),
        t_max=field("diffusion.eb_t_max", 0.8),
        entropy_bound=field("diffusion.eb_entropy_bound", 0.1),
        stability_threshold=field("diffusion.eb_stability_threshold", 1),
        confidence_threshold=field("diffusion.eb_confidence_threshold", 0.005),
    )


# --- the persistent logits server ------------------------------------------ #

class LogitsServer:
    """Drive ``llama-diffusion-gemma-server``: one model load, many forwards.

    Protocol (see the example's header): a request-file path per stdin line;
    the file holds ``int32 P, C, use_sc, float32 temp`` then ``P+C`` int32
    token ids (canvas last). The server replies ``C × n_vocab`` float32 canvas
    logits to ``<path>.resp`` and ``OK <C>`` on stdout. Self-conditioning
    state (the previous step's logits) is cached server-side.
    """

    def __init__(self, model_path: Path, *, ngl: int = 99, max_tokens: int = 4096,
                 kv_cache: bool = True, log_path: Path | None = None,
                 startup_timeout: float = 300.0):
        binary = llama_bin("llama-diffusion-gemma-server")  # raises if not built
        self._tmp = tempfile.TemporaryDirectory(prefix="dg-shim-")
        self._req_idx = 0
        self._lock = threading.Lock()
        env = {
            "NGL": str(ngl),
            "MAXTOK": str(max_tokens),
            "DG_KVCACHE": "1" if kv_cache else "0",
        }
        stderr: int | object = subprocess.DEVNULL
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr = log_path.open("w")
        self._proc = subprocess.Popen(
            [str(binary), str(model_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            env={**os.environ, **env}, text=True, bufsize=1,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        deadline = time.time() + startup_timeout
        line = ""
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-diffusion-gemma-server exited with code {self._proc.returncode} "
                    f"before READY (bad GGUF / unsupported arch?)"
                )
            line = self._stdout.readline()
            if line.startswith("READY"):
                break
        if not line.startswith("READY"):
            self.close()
            raise TimeoutError(f"logits server not READY within {startup_timeout}s")
        self.n_vocab = int(line.split()[1])

    def canvas_logits(
        self, prompt_ids: np.ndarray, canvas_ids: np.ndarray,
        *, use_sc: bool, temp: float,
    ) -> np.ndarray:
        """One forward; returns raw canvas logits ``[C, n_vocab]`` (float32)."""
        with self._lock:
            self._req_idx += 1
            req = Path(self._tmp.name) / f"req-{self._req_idx:06d}"
            header = np.empty(4, dtype=np.int32)
            header[0] = len(prompt_ids)
            header[1] = len(canvas_ids)
            header[2] = 1 if use_sc else 0
            header[3:4].view(np.float32)[0] = temp
            payload = np.concatenate([
                header,
                np.asarray(prompt_ids, dtype=np.int32),
                np.asarray(canvas_ids, dtype=np.int32),
            ])
            payload.tofile(req)
            self._stdin.write(f"{req}\n")
            self._stdin.flush()
            line = self._stdout.readline().strip()
            if not line.startswith("OK"):
                raise RuntimeError(f"logits server error: {line!r}")
            resp = Path(f"{req}.resp")
            logits = np.fromfile(resp, dtype=np.float32).reshape(
                len(canvas_ids), self.n_vocab
            )
            req.unlink(missing_ok=True)
            resp.unlink(missing_ok=True)
            return logits

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._stdin.write("QUIT\n")
                self._stdin.flush()
                self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        finally:
            self._tmp.cleanup()


# --- the entropy-bound sampler (numpy port) -------------------------------- #

def _detect_repetition_loop(tokens: list[int], start: int) -> bool:
    """llama.cpp trim_canvas loop rule: >= 6 consecutive stride-1/2 repeats."""
    n = len(tokens)
    for stride in (1, 2):
        reps = 0
        j = start
        while j + stride < n and tokens[j] == tokens[j + stride]:
            reps += 1
            j += stride
        if reps >= 6:
            return True
    return False


def trim_canvas(tokens: list[int], eog_ids: frozenset[int]) -> int:
    """Cut index for a denoised canvas: first EOG token or repetition loop."""
    cut = len(tokens)
    for i, t in enumerate(tokens):
        if t in eog_ids:
            cut = i
            break
    for i in range(max(cut - 1, 0)):
        if _detect_repetition_loop(tokens, i):
            cut = i
            break
    return cut


def entropy_bound_generate(
    server: LogitsServer,
    prompt_ids: list[int],
    *,
    eb: EbParams,
    eog_ids: frozenset[int],
    seed: int = 0,
    max_blocks: int = 4,
    max_tokens: int = 4096,
) -> list[int]:
    """Block-autoregressive entropy-bound denoising; returns generated ids.

    Faithful port of ``diffusion_generate_entropy_bound`` +
    ``run_turn``'s block loop from llama.cpp's diffusion example: each block
    denoises a fixed ``canvas_length`` canvas conditioned on the prefix, the
    trimmed result is committed and the next block starts, until an EOG
    token, a repetition loop, the block budget, or the token budget.
    """
    rng = np.random.default_rng(seed)
    C = eb.canvas_length
    S = max(1, eb.max_steps)
    prefix = list(prompt_ids)
    response: list[int] = []

    for _block in range(max_blocks):
        if len(prefix) + C > max_tokens:
            break
        canvas = rng.integers(0, server.n_vocab, size=C).astype(np.int64)
        argmax = np.full(C, -1, dtype=np.int64)
        prev_argmax = np.full(C, -2, dtype=np.int64)
        held = 0

        for step_idx in range(S):
            cur_step = S - step_idx
            t = eb.t_min + (eb.t_max - eb.t_min) * (cur_step / S)
            logits = server.canvas_logits(
                np.asarray(prefix), canvas, use_sc=step_idx > 0, temp=t,
            )
            z = logits.astype(np.float64) / t
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)

            argmax = logits.argmax(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                logp = np.where(p > 0, np.log(p), 0.0)
            entropy = -(p * logp).sum(axis=1)

            # one multinomial draw per position (inverse-CDF, like the C++)
            cum = p.cumsum(axis=1)
            u = rng.random(C)
            sampled = (cum >= u[:, None]).argmax(axis=1)

            # accept the lowest-entropy positions within the bound
            order = np.argsort(entropy, kind="stable")
            cum_e = np.concatenate([[0.0], entropy[order].cumsum()[:-1]])
            accepted = np.zeros(C, dtype=bool)
            accepted[order[cum_e <= eb.entropy_bound]] = True

            renoise = rng.integers(0, server.n_vocab, size=C)
            canvas = np.where(accepted, sampled, renoise).astype(np.int64)

            held = held + 1 if np.array_equal(argmax, prev_argmax) else 0
            prev_argmax = argmax
            confident = entropy.mean() < eb.confidence_threshold
            if held >= eb.stability_threshold and confident:
                break

        out = [int(x) for x in argmax]
        cut = trim_canvas(out, eog_ids)
        response.extend(out[:cut])
        if cut < C:
            break  # end token or repetition loop: answer complete
        prefix.extend(out[:cut])

    return response


# --- Gemma-4 response markup ------------------------------------------------ #

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:\s*([\w.\-]+)\s*(\{.*?\})\s*<tool_call\|>", re.S
)
_CHANNEL_RE = re.compile(r"<\|channel>.*?(?:<channel\|>|$)", re.S)
_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][\w\-]*)\s*:")


def _parse_gemma4_args(raw: str) -> dict:
    """Parse a gemma4-dict argument blob: JSON first, then with bare keys quoted."""
    for candidate in (raw, _BARE_KEY_RE.sub(r'\1"\2":', raw)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"_raw": raw}


def parse_gemma4_response(text: str) -> tuple[str, list[dict]]:
    """Split a raw Gemma-4 completion into (content, OpenAI-style tool_calls).

    The model emits ``<|tool_call>call:NAME{args}<tool_call|>`` segments and
    optional ``<|channel>…<channel|>`` reasoning blocks; both are stripped
    from the content.
    """
    tool_calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": m.group(1),
                "arguments": json.dumps(_parse_gemma4_args(m.group(2))),
            },
        })
    content = _TOOL_CALL_RE.sub("", text)
    content = _CHANNEL_RE.sub("", content)
    content = re.sub(r"<(?:end_of_turn|eos|\|im_end\|)>", "", content)
    return content.strip(), tool_calls


# --- the OpenAI-compatible HTTP shim ---------------------------------------- #

class _ShimState:
    def __init__(self, server: LogitsServer, tokenizer, eb: EbParams,
                 eog_ids: frozenset[int], max_blocks: int, max_tokens: int):
        self.server = server
        self.tokenizer = tokenizer
        self.eb = eb
        self.eog_ids = eog_ids
        self.max_blocks = max_blocks
        self.max_tokens = max_tokens
        self.model_name = "diffusion-gemma"


def _make_handler(state: _ShimState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request stderr noise
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler API
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self.path.endswith("/chat/completions"):
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length))
                messages = req["messages"]
                tools = req.get("tools") or None
                seed = int(req.get("seed", 0) or 0)

                prompt = state.tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    tokenize=False,
                )
                ids = state.tokenizer(
                    prompt, add_special_tokens=False
                ).input_ids
                out_ids = entropy_bound_generate(
                    state.server, ids, eb=state.eb, eog_ids=state.eog_ids,
                    seed=seed, max_blocks=state.max_blocks,
                    max_tokens=state.max_tokens,
                )
                raw = state.tokenizer.decode(out_ids, skip_special_tokens=False)
                content, tool_calls = parse_gemma4_response(raw)
                message: dict = {"role": "assistant", "content": content or None}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                self._json(200, {
                    "id": "dgshim",
                    "object": "chat.completion",
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }],
                    "usage": {
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(out_ids),
                        "total_tokens": len(ids) + len(out_ids),
                    },
                })
            except Exception as e:  # noqa: BLE001 — surfaced to the client
                self._json(500, {"error": {"message": repr(e)}})

    return Handler


def _gemma_eog_ids(tokenizer) -> frozenset[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok in ("<end_of_turn>",):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0:
            ids.add(int(tid))
    return frozenset(ids)


@contextmanager
def running_diffusion_server(
    model_path: Path,
    *,
    hf_dir: Path,
    ctx: int = 4096,
    ngl: int = 99,
    log_path: Path | None = None,
    startup_timeout: float = 300.0,
    max_blocks: int = 4,
    chat_template_kwargs: str | None = None,  # accepted for interface parity; unused
):
    """Drop-in sibling of :func:`quant_tuner.eval.server.running_server` for
    DiffusionGemma GGUFs: yields a ``base_url`` whose ``/v1/chat/completions``
    runs block-diffusion generation. ``hf_dir`` must hold the HF tokenizer
    (chat template + tools rendering) — the extracted snapshot
    (``ws.model_extracted``) is the canonical source.
    """
    del chat_template_kwargs
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    eb = read_eb_params(model_path)
    logits_server = LogitsServer(
        model_path, ngl=ngl, max_tokens=ctx,
        log_path=log_path, startup_timeout=startup_timeout,
    )
    try:
        state = _ShimState(
            logits_server, tokenizer, eb,
            _gemma_eog_ids(tokenizer), max_blocks, ctx,
        )
        port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
    finally:
        logits_server.close()


__all__ = [
    "EbParams",
    "LogitsServer",
    "entropy_bound_generate",
    "parse_gemma4_response",
    "read_eb_params",
    "running_diffusion_server",
    "trim_canvas",
]
