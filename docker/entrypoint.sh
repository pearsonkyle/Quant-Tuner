#!/bin/bash
# Entrypoint for the red-team eval container.
# Verifies the source mount, then drops to the unprivileged user via gosu.
set -e

USER_NAME="${MY_USERNAME:-appuser}"

cd /project || {
    echo "ERROR: /project not found — is the source bind-mount configured?" >&2
    exit 1
}

if [ ! -d "/project/src/quant_tuner/eval" ]; then
    echo "ERROR: /project/src/quant_tuner/eval not found." >&2
    echo "       Mount the quant-tuner repo root at /project (see run.sh)." >&2
    ls -la /project >&2
    exit 1
fi

# If a Docker socket is mounted, make sure the user can use it (parity with the
# base image; harmless if absent).
if [ -S /var/run/docker.sock ]; then
    DOCKER_SOCK_GID=$(stat -c '%g' /var/run/docker.sock)
    if ! getent group "${DOCKER_SOCK_GID}" > /dev/null 2>&1; then
        groupadd -g "${DOCKER_SOCK_GID}" docker 2>/dev/null || true
    fi
    usermod -aG "${DOCKER_SOCK_GID}" "${USER_NAME}" 2>/dev/null || true
fi

echo "Container started with: $*"
exec gosu "${USER_NAME}" "$@"
