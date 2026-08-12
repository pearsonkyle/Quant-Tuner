#!/bin/bash
# Run the red-team eval container.
#
# Mounts the quant-tuner repo at /project and runs the given command inside the
# slim image as the host user. The GAIAA-API cert handling is borrowed from
# LLM-Training-and-Quantization/docker/run.sh: the host CA bundle is mounted over
# the container's, and REQUESTS_CA_BUNDLE / SSL_CERT_FILE point at it so HTTPS to
# https://api.ai.gd-ms.us works once GAIAA_API_KEY is set in the environment.
#
# Usage:
#   docker/run.sh                                   # interactive bash
#   docker/run.sh bash scripts/redteam_compare.sh   # run the comparison
#   docker/run.sh python scripts/eval_redteam.py --help
#
# Pass GAIAA_API_KEY in your shell env to use the GAIAA API as a judge/target:
#   GAIAA_API_KEY=sk-... docker/run.sh bash scripts/redteam_compare.sh
set -euo pipefail

current_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
repo_root=$(readlink -f "${current_dir}/..")

source "${current_dir}/.env"
source "${current_dir}/gid.env"

# Default to an interactive shell if no command is given.
if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi

CONTAINER_NAME="${CONTAINER_NAME:-${MY_USERNAME}-redteam}"

# Allocate a TTY only when stdin is one (so piped / CI invocations still work).
TTY_FLAGS=(-i)
[ -t 0 ] && TTY_FLAGS+=(-t)

# Host CA bundle (RHEL/CentOS path on this host; falls back to the Debian path).
HOST_CA_BUNDLE="/etc/ssl/certs/ca-bundle.crt"
[ -f "${HOST_CA_BUNDLE}" ] || HOST_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

DOCKER_RUN_ARGS=(
    --rm "${TTY_FLAGS[@]}"
    --name "${CONTAINER_NAME}"
    --env-file "${current_dir}/.env"
    --env "MY_USERNAME=${MY_USERNAME}"
    # Source mount.
    -v "${repo_root}:/project"
    # Live entrypoint (so edits don't require a rebuild).
    -v "${current_dir}/entrypoint.sh:/usr/local/bin/entrypoint.sh:ro"
    # ── GAIAA cert handling (borrowed from the base image) ──────────────────
    -v "${HOST_CA_BUNDLE}:/etc/ssl/certs/ca-certificates.crt:ro"
    --env "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
    # ── API keys (GAIAA mapped to OPENAI_* like the base image) ─────────────
    --env "GAIAA_API_KEY=${GAIAA_API_KEY:-}"
    --env "OPENAI_API_KEY=${OPENAI_API_KEY:-${GAIAA_API_KEY:-}}"
    # Data roots so on-network model/output paths resolve if ever needed.
    -v /data/gondor:/data/gondor
)
[ -n "${NETWORK:-}" ] && DOCKER_RUN_ARGS+=(--network "${NETWORK}")

echo "Running: docker run ... ${IMAGE_NAME} $*"
docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE_NAME}" "$@"
