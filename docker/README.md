# Red-team eval container

A minimal, CPU-only container for running the red-team safety eval
(`scripts/eval_redteam.py`, `scripts/redteam_compare.sh`) as an API client.

The eval only talks to remote OpenAI-compatible endpoints (target / judge /
simulator), so this image is built on `python:3.12-slim` and installs **only**
the red-team deps (`deepteam`, `deepeval`, `openai`, `pyyaml`) — no torch, no
llama.cpp, no CUDA. The repo source is bind-mounted at `/project` and run via
`PYTHONPATH=src`, so the heavy core deps in `pyproject.toml` are never installed.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | slim base, non-root user (from `gid.env`), red-team deps |
| `requirements_redteam.txt` | pinned deps (matched to repo `uv.lock`) |
| `entrypoint.sh` | verifies the source mount, drops to the host user via `gosu` |
| `build.sh` | builds the image (ported from the base-image build script) |
| `run.sh` | runs a command in the container with GAIAA cert + key wiring |
| `.env` / `gid.env` | image name, base image, GAIAA settings, host uid/gid |

## Build

```bash
docker/build.sh
```

Produces the image tagged `${MY_USERNAME}-quant-tuner-redteam`.

## Run

```bash
# Interactive shell
docker/run.sh

# Run the gemma-vs-gemma comparison (judge = minimax on the local network).
# PY=python uses the image's system deps instead of `uv run` (which would
# recreate the full project venv, pulling torch/CUDA — not what we want here).
docker/run.sh bash -lc 'PY=python CONFIG=red_team_minimal bash scripts/redteam_compare.sh'

# One-off single-target run
docker/run.sh python scripts/eval_redteam.py --help
```

`run.sh` bind-mounts the repo at `/project`, runs as your host user, and mounts
`/data/gondor` so on-network paths resolve.

## GAIAA API

Cert handling mirrors the `LLM-Training-and-Quantization` image: the host CA
bundle is mounted over the container's, and `REQUESTS_CA_BUNDLE` /
`SSL_CERT_FILE` point at it, so HTTPS to `https://api.ai.gd-ms.us` validates.

Provide your key in the shell; `run.sh` maps `GAIAA_API_KEY` → `OPENAI_API_KEY`
inside the container:

```bash
GAIAA_API_KEY=sk-... docker/run.sh python scripts/eval_redteam.py \
    --base-url http://10.132.212.38:10201/v1 --target-model-name gemma-4-31b \
    --config red_team_minimal \
    --judge-model claude-4-6-sonnet --judge-base-url https://api.ai.gd-ms.us/v1 \
    --judge-api-key "$GAIAA_API_KEY" \
    --simulator-model claude-4-6-sonnet --simulator-base-url https://api.ai.gd-ms.us/v1 \
    --simulator-api-key "$GAIAA_API_KEY" \
    --out out/redteam/gaiaa_judge.csv
```

To use GAIAA as the **target**, pass its `--base-url` / `--target-model-name`
and `--api-key "$GAIAA_API_KEY"` instead.

## Notes

- The image is single-purpose; deps are installed system-wide (no venv).
- Telemetry is disabled (`DEEPEVAL_TELEMETRY_OPT_OUT`, `DO_NOT_TRACK`).
- `gid.env` is checked in with the current user's uid/gid — regenerate it
  (`id -u`, `id -g`, `whoami`) if another user builds the image.
