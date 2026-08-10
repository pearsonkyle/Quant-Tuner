# Minimal red-team eval container.
#
# This is an API-only client (target / judge / simulator are remote
# OpenAI-compatible endpoints), so we use a slim Python base and install just
# the red-team deps — no torch, no llama.cpp, no CUDA. The quant-tuner source is
# bind-mounted at runtime (see run.sh) and run via PYTHONPATH=src, so the heavy
# core deps in pyproject.toml are never installed.
#
# Certs for the GAIAA API are handled at RUNTIME (run.sh mounts the host CA
# bundle over /etc/ssl/certs/ca-certificates.crt and sets REQUESTS_CA_BUNDLE /
# SSL_CERT_FILE) — the same pattern as the LLM-Training-and-Quantization image.
# We still install ca-certificates here so the bundle path exists to mount onto.
ARG BUILD_IMAGE=python:3.12-slim
FROM ${BUILD_IMAGE}

ARG MY_USERNAME=appuser
ARG MY_UID=1000
ARG MY_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    DO_NOT_TRACK=1

# gosu for clean privilege drop in the entrypoint; ca-certificates so the bundle
# mount target exists; curl for health checks; git in case a dep needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gosu ca-certificates curl git tini && \
    update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Match the host user (gid.env) so bind-mounted files keep correct ownership.
RUN groupadd -g ${MY_GID} ${MY_USERNAME} && \
    useradd -m -u ${MY_UID} -g ${MY_GID} -s /bin/bash ${MY_USERNAME}

WORKDIR /project

# uv for fast, lock-consistent installs.
RUN pip install --upgrade uv

# Install the red-team client deps system-wide (image is single-purpose).
COPY requirements_redteam.txt /tmp/requirements_redteam.txt
RUN uv pip install --system -r /tmp/requirements_redteam.txt && \
    rm /tmp/requirements_redteam.txt

# Source is bind-mounted at /project at runtime; scripts run with PYTHONPATH=src.
ENV PYTHONPATH=/project/src

COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# tini reaps zombies; entrypoint drops to the unprivileged user via gosu.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
