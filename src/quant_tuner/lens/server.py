# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from igorbarshteyn/jlens-gguf (Apache-2.0) for quant-tuner; see NOTICE.
"""The jlens bridge: web UI + JSON API between the browser and jlens-server(s).

Adapted from the upstream single-backend bridge, extended for quant-tuner with:

- an optional **second backend** (``--model-b``): the same live-slice API works
  against backend A or B (``?backend=b``), so two quants can sit side by side;
- a **capture-run store** so slices persist to disk as
  :class:`quant_tuner.lens.capture.Run` directories, and a
  **diff** endpoint (``/api/diff``) that compares any two stored runs
  (top-1 disagreement, ref-top1 rank shift, per-layer + per-position KLD) —
  the data behind the A/B heatmap;
- a **save-direction** endpoint that persists a UI intervention as a
  ``direction.npz`` for :mod:`quant_tuner.lens.bake`.

The core single-backend slice/ranks/readout/decompose/generate API is kept
compatible with the copied ``web/app.js`` so the base visualizer works
unchanged; the A/B additions live under new ``/api/*`` routes consumed by
``web/ab.js`` and ``web/diff_heatmap.js``.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from quant_tuner.lens.client import NativeClient
from quant_tuner.lens.lens import JacobianLensGGUF
from quant_tuner.lens.model_reader import ReadoutWeights
from quant_tuner.lens.readout import LensReadout, compute_grid

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def safe_static_path(rel: str, root: Path = WEB_DIR) -> Path | None:
    web_root = root.resolve()
    target = (web_root / rel.lstrip("/")).resolve()
    if target == web_root or web_root in target.parents:
        return target
    return None


class LogitsCache:
    """LRU cache of per-(ctx, layer) lens logits, bounded by bytes."""

    def __init__(self, budget_bytes: int = 1 << 30) -> None:
        self.budget = budget_bytes
        self._data: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(self, key: tuple) -> np.ndarray | None:
        arr = self._data.get(key)
        if arr is not None:
            self._data.move_to_end(key)
        return arr

    def put(self, key: tuple, arr: np.ndarray) -> None:
        if key in self._data:
            self._bytes -= self._data.pop(key).nbytes
        self._data[key] = arr
        self._bytes += arr.nbytes
        while self._bytes > self.budget and len(self._data) > 1:
            _, old = self._data.popitem(last=False)
            self._bytes -= old.nbytes

    def drop_ctx(self, ctx_id: str) -> None:
        for key in [k for k in self._data if k[0] == ctx_id]:
            self._bytes -= self._data.pop(key).nbytes


class Context:
    """One computed slice: tokens, activations, grid."""

    def __init__(self, ctx_id, tokens, n_prompt, n_gen, activations, grid, use_lens, interventions):
        self.ctx_id = ctx_id
        self.tokens = tokens
        self.n_prompt = n_prompt
        self.n_gen = n_gen
        self.activations = activations
        self.grid = grid
        self.use_lens = use_lens
        self.interventions = interventions


class Backend:
    """A single model+lens served by one jlens-server, with slice bookkeeping."""

    def __init__(
        self,
        *,
        name: str,
        model_path: str,
        native_url: str,
        lens_path: str | None = None,
        top_n: int = 10,
        max_contexts: int = 3,
        logits_cache_mb: int = 1024,
    ) -> None:
        self.name = name
        self.client = NativeClient(native_url)
        self.props = self.client.props()
        self.n_layers = self.props["n_layer"]
        self.model_path = model_path

        self.weights = ReadoutWeights.from_gguf(model_path)
        if self.weights.n_layers != self.n_layers:
            raise ValueError("model GGUF disagrees with native server on layer count")

        if lens_path:
            self.lens = JacobianLensGGUF.load(lens_path)
        else:
            self.lens = JacobianLensGGUF.identity(
                d_model=self.weights.d_model,
                layers=list(range(self.n_layers - 1)),
                base_model=model_path,
            )
        self.lens_path = lens_path
        self.readout = LensReadout(self.weights, self.lens)

        self.vocab = self.client.vocab()
        self.vocab_attrs = self.client.vocab_attrs()
        self.vocab_lower = [p.lower() for p in self.vocab]

        self.top_n = top_n
        self.max_tokens = self.props["n_ctx"] - 8
        self.contexts: OrderedDict[str, Context] = OrderedDict()
        self.max_contexts = max_contexts
        self.logits_cache = LogitsCache(logits_cache_mb << 20)
        self.lock = threading.Lock()
        self._ctx_counter = 0

    # ---- helpers (unchanged from upstream) ----

    def pieces(self, tokens: list[int]) -> list[str]:
        return [self.vocab[t] if 0 <= t < len(self.vocab) else f"[{t}]" for t in tokens]

    def resolve_tokens(self, body: dict) -> list[int]:
        if body.get("tokens"):
            tokens = [int(t) for t in body["tokens"]]
        elif body.get("messages"):
            prompt = self.client.apply_template(body["messages"], add_assistant=True)
            if body.get("assistant_prefill"):
                prompt += body["assistant_prefill"]
            tokens = self.client.tokenize(prompt, add_special=True, parse_special=True)
        elif body.get("prompt") is not None:
            tokens = self.client.tokenize(
                body["prompt"],
                add_special=bool(body.get("add_special", True)),
                parse_special=bool(body.get("parse_special", True)),
            )
        else:
            raise ValueError("provide 'prompt', 'tokens', or 'messages'")
        if not tokens:
            raise ValueError("empty prompt")
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return tokens

    def readout_layers(self, stride: int = 1) -> list[int]:
        layers = self.readout.readout_layers(self.n_layers)
        if stride > 1:
            final = layers[-1]
            layers = layers[::stride]
            if final not in layers:
                layers.append(final)
        return layers

    def translate_interventions(self, specs: list[dict], tokens: list[int] | None) -> list[dict]:
        if not specs:
            return []
        fitted = set(self.lens.source_layers)
        native: list[dict] = []

        def layer_range(spec) -> list[int]:
            lr = spec.get("layers")
            if lr is None:
                return sorted(fitted)
            l0, l1 = int(lr[0]), int(lr[1])
            return [l for l in sorted(fitted) if l0 <= l <= l1]

        steer_layers = sorted(
            {l for s in specs if s.get("type") == "steer" for l in layer_range(s)}
        )
        norms = {l: self.lens.h_rms.get(l, 1.0) for l in steer_layers}

        for spec in specs:
            kind = spec.get("type")
            pos = spec.get("pos") or [0, -1]
            p0, p1 = int(pos[0]), int(pos[1])
            if kind == "steer":
                tid = int(spec["token_id"])
                alpha = float(spec.get("alpha", 2.0))
                for l in layer_range(spec):
                    vec = self.readout.steer_vector(l, tid, alpha, h_rms=norms.get(l))
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "add", "vector": vec})
            elif kind == "ablate":
                tid = int(spec["token_id"])
                for l in layer_range(spec):
                    A, B = self.readout.ablate_factors(l, [tid])
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            elif kind == "swap":
                ta, tb = int(spec["token_a"]), int(spec["token_b"])
                for l in layer_range(spec):
                    A, B = self.readout.swap_factors(l, ta, tb)
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            else:
                raise ValueError(f"unknown intervention type {kind!r}")
        return native

    def lens_logits_for(self, ctx: Context, layer: int) -> np.ndarray:
        key = (ctx.ctx_id, layer, ctx.use_lens)
        cached = self.logits_cache.get(key)
        if cached is not None:
            return cached
        logits = self.readout.lens_logits(ctx.activations[layer], layer, use_lens=ctx.use_lens)
        self.logits_cache.put(key, logits.astype(np.float32))
        return logits

    def _ctx(self, body: dict) -> Context:
        ctx = self.contexts.get(str(body.get("ctx_id")))
        if ctx is None:
            raise ValueError("unknown or expired ctx_id; re-run the slice")
        return ctx

    def props_json(self) -> dict:
        return {
            "name": self.name,
            "model": self.props,
            "model_name": self.weights.model_name or Path(self.props["model_path"]).name,
            "arch": self.weights.arch,
            "n_vocab": self.weights.n_vocab,
            "d_model": self.weights.d_model,
            "n_layers": self.n_layers,
            "lens": {
                "path": self.lens_path,
                "method": self.lens.fit_method,
                "n_prompts": self.lens.n_prompts,
                "source_layers": self.lens.source_layers,
                "target_layer": self.lens.target_layer,
                "has_bias": bool(self.lens.biases),
            },
            "top_n": self.top_n,
        }

    def api_slice(self, body: dict) -> dict:
        t0 = time.time()
        tokens = self.resolve_tokens(body)
        use_lens = bool(body.get("use_lens", True))
        top_n = int(body.get("top_n", self.top_n))
        stride = max(1, int(body.get("stride", 1)))
        n_predict = int(body.get("n_predict", 0))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []

        layers = self.readout_layers(stride)
        native_ivs = self.translate_interventions(ui_specs, tokens)
        fr = self.client.forward(
            tokens, capture_layers=layers, dtype="f16",
            interventions=native_ivs, n_predict=n_predict, sampling=sampling,
        )
        t_fwd = time.time()

        self._ctx_counter += 1
        ctx_id = f"{self.name}{self._ctx_counter}"
        ctx = Context(ctx_id, fr.tokens, fr.n_prompt, fr.n_gen, fr.activations, None, use_lens, ui_specs)
        grid = compute_grid(
            self.readout, fr.activations, layers, top_n=top_n, use_lens=use_lens,
            logits_fn=lambda layer: self.lens_logits_for(ctx, layer),
        )
        ctx.grid = grid
        self.contexts[ctx_id] = ctx
        while len(self.contexts) > self.max_contexts:
            old_id, _ = self.contexts.popitem(last=False)
            self.logits_cache.drop_ctx(old_id)

        return {
            "backend": self.name,
            "ctx_id": ctx_id,
            "tokens": ctx.tokens,
            "pieces": self.pieces(ctx.tokens),
            "n_prompt": ctx.n_prompt,
            "n_gen": ctx.n_gen,
            "generated_text": fr.generated_text,
            "layers": layers,
            "top_n": top_n,
            "use_lens": use_lens,
            "top_ids": _b64(grid.top_ids.astype("<i4")),
            "norms": {str(l): v for l, v in grid.norms.items()},
            "vocab_size": self.weights.n_vocab,
            "interventions": ui_specs,
            "timings": {
                "forward_ms": round((t_fwd - t0) * 1000, 1),
                **fr.timings,
            },
        }

    def api_ranks(self, body: dict) -> dict:
        ctx = self._ctx(body)
        token_ids = np.asarray([int(t) for t in body["token_ids"]], dtype=np.int64)
        T = len(ctx.tokens)
        L = len(ctx.grid.layers)
        out = np.zeros((T, L, len(token_ids)), dtype=np.int32)
        for li, layer in enumerate(ctx.grid.layers):
            logits = self.lens_logits_for(ctx, layer)
            out[:, li, :] = LensReadout.ranks_of(logits, token_ids)
        return {
            "ctx_id": ctx.ctx_id, "token_ids": [int(t) for t in token_ids],
            "shape": [T, L, len(token_ids)], "ranks": _b64(out.astype("<i4")),
        }

    def api_readout(self, body: dict) -> dict:
        ctx = self._ctx(body)
        pos = int(body["pos"])
        layer = int(body["layer"])
        top_n = int(body.get("top_n", 40))
        logits = self.lens_logits_for(ctx, layer)[pos]
        ids, vals = LensReadout.topk(logits[None, :], top_n)
        ids, vals = ids[0], vals[0]
        z = logits - logits.max()
        probs_all = np.exp(z)
        probs_all /= probs_all.sum()
        return {
            "pos": pos, "layer": layer,
            "tokens": [
                {"token": int(t), "piece": self.vocab[int(t)],
                 "logit": float(v), "prob": float(probs_all[int(t)])}
                for t, v in zip(ids, vals, strict=False)
            ],
        }

    def api_decompose(self, body: dict) -> dict:
        ctx = self._ctx(body)
        pos = int(body["pos"])
        layer = int(body["layer"])
        k = min(int(body.get("k", 12)), 25)
        if layer not in self.lens.jacobians and layer != self.n_layers - 1:
            raise ValueError(f"layer {layer} not fitted")
        h = ctx.activations[layer][pos]
        items = self.readout.decompose(h, layer, k=k)
        for item in items:
            item["piece"] = self.vocab[item["token"]]
        return {"pos": pos, "layer": layer, "items": items}

    def api_generate(self, body: dict) -> dict:
        if body.get("ctx_id"):
            ctx = self._ctx(body)
            tokens = ctx.tokens[: ctx.n_prompt]
        else:
            tokens = self.resolve_tokens(body)
        n_predict = int(body.get("n_predict", 32))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []
        native_ivs = self.translate_interventions(ui_specs, tokens)
        fr = self.client.forward(tokens, capture=False, interventions=native_ivs,
                                 n_predict=n_predict, sampling=sampling)
        out = {"steered": {"text": fr.generated_text, "tokens": [g["token"] for g in fr.generated]}}
        if body.get("compare", True) and ui_specs:
            fr0 = self.client.forward(tokens, capture=False, n_predict=n_predict, sampling=sampling)
            out["baseline"] = {"text": fr0.generated_text, "tokens": [g["token"] for g in fr0.generated]}
        return out

    def api_search_tokens(self, query: str, limit: int = 50) -> dict:
        q = query.strip()
        results: list[tuple[int, int]] = []
        if q.startswith("#") and q[1:].isdigit():
            tid = int(q[1:])
            if 0 <= tid < len(self.vocab):
                results.append((0, tid))
        ql = q.lower()
        if ql:
            for tid, piece in enumerate(self.vocab_lower):
                stripped = piece.strip()
                if not stripped:
                    continue
                if stripped == ql:
                    results.append((1, tid))
                elif stripped.startswith(ql):
                    results.append((2, tid))
                elif ql in stripped:
                    results.append((3, tid))
        results.sort(key=lambda x: (x[0], len(self.vocab[x[1]]), x[1]))
        return {"results": [{"token": tid, "piece": self.vocab[tid]} for _, tid in results[:limit]]}


class App:
    """The bridge: one or two backends + a capture-run store + diff engine."""

    def __init__(
        self,
        *,
        backend_a: Backend,
        backend_b: Backend | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self.a = backend_a
        self.b = backend_b
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.lock = threading.Lock()

    def backend(self, body_or_query) -> Backend:
        name = None
        if isinstance(body_or_query, dict):
            name = body_or_query.get("backend")
        if name == "b":
            if self.b is None:
                raise ValueError("no second backend (--model-b) is configured")
            return self.b
        return self.a

    # ---- A/B additions ----

    def api_backends(self) -> dict:
        return {
            "a": self.a.props_json(),
            "b": self.b.props_json() if self.b else None,
            "runs_dir": str(self.runs_dir) if self.runs_dir else None,
        }

    def api_runs(self, query: dict) -> dict:
        if self.runs_dir is None:
            return {"runs": []}
        from quant_tuner.lens.capture import find_runs

        tags = {k: v[0] for k, v in query.items() if k not in ("_",)}
        runs = find_runs(self.runs_dir, **tags)
        return {
            "runs": [
                {"run_id": m.run_id, "model": Path(m.model_path).name,
                 "layers": len(m.capture_layers), "positions": len(m.positions),
                 "n_gen": m.n_gen, "tags": m.tags, "created_at": m.created_at}
                for m in runs
            ]
        }

    def api_capture(self, body: dict) -> dict:
        if self.runs_dir is None:
            raise ValueError("capture store not configured (pass --runs-dir)")
        from quant_tuner.lens.capture import capture_run

        backend = self.backend(body)
        tokens = backend.resolve_tokens(body)
        specs = body.get("interventions") or []
        native = backend.translate_interventions(specs, tokens)
        tail = int(body.get("tail", 0))
        keep = list(range(-tail, 0)) if tail > 0 else None
        run_dir = capture_run(
            backend.client, tokens, runs_dir=self.runs_dir,
            keep_positions=keep, interventions=specs, native_interventions=native,
            n_predict=int(body.get("n_predict", 0)),
            sampling=body.get("sampling") or {"greedy": True},
            logits_positions=[-1],
            tags={**(body.get("tags") or {}), "model": backend.weights.model_name or backend.name},
        )
        return {"run_id": run_dir.name}

    def api_diff(self, query: dict) -> dict:
        if self.runs_dir is None:
            raise ValueError("capture store not configured")
        from quant_tuner.lens.capture import load_run
        from quant_tuner.lens.diff import diff_runs

        a_id = query["a"][0]
        b_id = query["b"][0]
        pins = [int(x) for x in query.get("pin", [""])[0].split(",") if x.strip()]
        run = load_run(self.runs_dir / a_id)
        ref = load_run(self.runs_dir / b_id)
        # pick each run's backend readout by model path
        readout_run = self._readout_for(run.manifest.model_path)
        readout_ref = self._readout_for(ref.manifest.model_path)
        d = diff_runs(run, ref, readout_run=readout_run, readout_ref=readout_ref,
                      pinned_tokens=pins or None)
        return {
            "meta": d.to_meta(),
            "agree": _b64(d.agree.astype(np.uint8)),
            "ref_top1_rank_in_run": _b64(d.ref_top1_rank_in_run.astype("<i4")),
            "top1_run": _b64(d.top1_run.astype("<i4")),
            "top1_ref": _b64(d.top1_ref.astype("<i4")),
            "layer_kld": [float(x) for x in d.layer_kld],
            "layer_rel_l2": [float(x) for x in d.layer_rel_l2],
            "final_kld": [float(x) for x in d.final_kld],
            "final_kld_exact": [bool(x) for x in d.final_kld_exact],
            "shape": [len(d.layers), len(d.positions)],
            "pinned": {
                str(t): {"rank_run": _b64(v["rank_run"].astype("<i4")),
                         "rank_ref": _b64(v["rank_ref"].astype("<i4"))}
                for t, v in d.pinned.items()
            },
        }

    def api_save_direction(self, body: dict) -> dict:
        if self.runs_dir is None:
            raise ValueError("capture store not configured")
        from quant_tuner.lens.bake import Direction

        backend = self.backend(body)
        spec = body["intervention"]
        tid = int(spec["token_id"])
        layer = int(spec.get("layer", backend.lens.source_layers[len(backend.lens.source_layers) // 2]))
        v = backend.readout.lens_vector(layer, tid, unit=True)
        name = body.get("name", f"dir_{tid}_L{layer}")
        out = self.runs_dir / "directions" / f"{name}.npz"
        Direction(v=v, layers=[layer], method="ui_lens_vector",
                  source={"token_id": tid, "layer": layer}).save(out)
        return {"path": str(out)}

    def _readout_for(self, model_path: str) -> LensReadout:
        for backend in (self.a, self.b):
            if backend is not None and Path(backend.model_path).resolve() == Path(model_path).resolve():
                return backend.readout
        # fall back: build a readout from the model + backend A's lens
        weights = ReadoutWeights.from_gguf(model_path)
        return LensReadout(weights, self.a.lens)


# ---------------------------------------------------------------------- #
# HTTP plumbing
# ---------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]  # set by serve_app() before listen
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            app = self.app
            if url.path == "/api/backends":
                return self._send_json(app.api_backends())
            if url.path == "/api/props":
                q = parse_qs(url.query)
                return self._send_json(app.backend({"backend": q.get("backend", ["a"])[0]}).props_json())
            if url.path == "/api/runs":
                return self._send_json(app.api_runs(parse_qs(url.query)))
            if url.path == "/api/diff":
                return self._send_json(app.api_diff(parse_qs(url.query)))
            if url.path == "/api/vocab":
                return self._send_json({"pieces": app.a.vocab, "attrs": app.a.vocab_attrs})
            if url.path == "/api/search_tokens":
                qs = parse_qs(url.query)
                backend = app.backend({"backend": qs.get("backend", ["a"])[0]})
                return self._send_json(backend.api_search_tokens(
                    qs.get("q", [""])[0], int(qs.get("limit", ["50"])[0])))
            rel = url.path.lstrip("/") or "index.html"
            target = safe_static_path(rel)
            if target is None:
                return self._send_json({"error": "forbidden"}, 403)
            return self._send_file(target)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            logger.exception("GET %s failed", self.path)
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            url = urlparse(self.path)
            app = self.app
            if url.path == "/api/capture":
                with app.lock:
                    return self._send_json(app.api_capture(body))
            if url.path == "/api/save_direction":
                with app.lock:
                    return self._send_json(app.api_save_direction(body))
            backend = app.backend(body)
            if url.path == "/api/tokenize":
                toks, pieces = backend.client.tokenize_with_pieces(body["content"])
                return self._send_json({"tokens": toks, "pieces": pieces})
            if url.path == "/api/template":
                prompt = backend.client.apply_template(
                    body["messages"], add_assistant=bool(body.get("add_assistant", True)))
                return self._send_json({"prompt": prompt})
            routes = {
                "/api/slice": backend.api_slice,
                "/api/ranks": backend.api_ranks,
                "/api/readout": backend.api_readout,
                "/api/decompose": backend.api_decompose,
                "/api/generate": backend.api_generate,
            }
            fn = routes.get(url.path)
            if fn is None:
                return self._send_json({"error": "not found"}, 404)
            with backend.lock:
                return self._send_json(fn(body))
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": str(e)}, 500)


def serve_app(
    *,
    model: Path,
    lens_path: Path | None,
    model_b: Path | None = None,
    lens_b: Path | None = None,
    runs_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8090,
    ctx: int = 8192,
    ngl: int = 99,
) -> None:
    """Spawn backend server(s), build the App, and serve the visualizer."""
    from contextlib import ExitStack

    from quant_tuner.lens.backend import running_jlens_server

    with ExitStack() as stack:
        client_a = stack.enter_context(running_jlens_server(model, ctx=ctx, ngl=ngl))
        backend_a = Backend(
            name="a", model_path=str(model), native_url=client_a.base_url,
            lens_path=str(lens_path) if lens_path else None,
        )
        backend_b = None
        if model_b is not None:
            client_b = stack.enter_context(running_jlens_server(model_b, ctx=ctx, ngl=ngl))
            backend_b = Backend(
                name="b", model_path=str(model_b), native_url=client_b.base_url,
                lens_path=str(lens_b or lens_path) if (lens_b or lens_path) else None,
            )
        app = App(backend_a=backend_a, backend_b=backend_b, runs_dir=runs_dir)
        Handler.app = app
        httpd = ThreadingHTTPServer((host, port), Handler)
        print(f"\n  ── J-Lens visualizer: http://{host}:{port} "
              f"({'A/B' if backend_b else 'single'}) ──\n", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
