# jlens-server

A llama-server-compatible activation server for the Jacobian lens, adapted
from [igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf)
(Apache-2.0; see `LICENSE` and the repo-root `NOTICE`).

It serves a GGUF with llama.cpp, intercepts each layer's residual output
(`l_out-<il>`) through the `ggml_backend_sched` eval callback, applies
runtime interventions (`add` steering / `lowrank` ablate-swap / `set`)
mid-graph, and exposes:

- `POST /jlens/forward` — activation capture (binary `JLNS` response) with
  optional interventions, generation, and full-vocab logits at requested
  positions;
- `GET/POST/DELETE /jlens/interventions` + `GET /jlens/last_completion` —
  a live intervention set applied to every completion;
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` — so it is a
  drop-in llama-server replacement: point any agent framework at it and steer
  live;
- `/props`, `/vocab`, `/tokenize`, `/detokenize`, `/apply_template`, `/health`.

## Build

The source is tracked here (not inside the llama.cpp submodule — unlike the
older `llama-mtp-capture` tool, a submodule bump can never silently drop it)
and links only llama.cpp's public API:

```bash
# uses the repo's vendored llama.cpp build (Metal/CUDA/CPU — whatever you built)
bash native/jlens_server/build.sh

# or point at another checkout (same env var paths.py honors)
LLAMA_CPP_DIR=/path/to/llama.cpp bash native/jlens_server/build.sh

# no llama.cpp build yet and you just want CPU libraries:
bash native/jlens_server/build.sh --auto-build
```

Or via the CLI: `uv run quant-tuner lens build-server`.

## Architecture support

At startup the server decodes one probe token and verifies it captured every
layer's `l_out` tensor; the result is reported as `l_out_ok` in `/props` and
asserted by the Python client before any capture. `parse_l_out` accepts
backend-split tensor names (`CUDA0#l_out-12#0`), so GPU offload works.
The llama.cpp commit the binary was built against is stamped into `/props`
(`llama_commit`) and recorded in every capture-run manifest, guarding A/B
comparisons across submodule bumps.
