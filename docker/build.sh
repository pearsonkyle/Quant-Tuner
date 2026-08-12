#!/bin/bash
# Builds the minimal red-team eval image.
# Ported from LLM-Training-and-Quantization/docker/base/build.sh, adapted for the
# single slim Dockerfile in this directory (no MODEL_CACHE / GPU build args).
set -euo pipefail

current_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Shared definitions.
source "${current_dir}/.env"
# Per-user uid/gid/username (so bind-mounted files keep correct ownership).
source "${current_dir}/gid.env"

echo "Building red-team image: ${IMAGE_NAME}"
echo "  base image : ${BUILD_IMAGE}"
echo "  user       : ${MY_USERNAME} (${MY_UID}:${MY_GID})"

docker build --pull --rm \
  -f "${current_dir}/Dockerfile" \
  -t "${IMAGE_NAME}" "${current_dir}" \
  --build-arg BUILD_IMAGE="${BUILD_IMAGE}" \
  --build-arg MY_USERNAME="${MY_USERNAME}" \
  --build-arg MY_UID="${MY_UID}" \
  --build-arg MY_GID="${MY_GID}"

echo "Done. Image: ${IMAGE_NAME}"
