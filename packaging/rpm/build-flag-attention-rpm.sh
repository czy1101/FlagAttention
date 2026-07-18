#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
# Defaults preserve the original single-arg Fedora flow:
#   ./build-flag-attention-rpm.sh [fedora-version]
# Other distros via env, e.g. openEuler 24.03:
#   BASE_IMAGE=openeuler/openeuler BASE_IMAGE_VERSION=24.03-lts ./build-flag-attention-rpm.sh
BASE_IMAGE="${BASE_IMAGE:-fedora}"
BASE_IMAGE_VERSION="${BASE_IMAGE_VERSION:-${1:-43}}"
if [ "${BASE_IMAGE}" = "fedora" ]; then
    IMAGE_TAG="flag-attention-rpm:f${BASE_IMAGE_VERSION}"
else
    IMAGE_TAG="flag-attention-rpm:$(echo "${BASE_IMAGE}-${BASE_IMAGE_VERSION}" | tr '/' '-')"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/rpm-packages}"

docker build --network=host \
    -f "${SCRIPT_DIR}/dockerfiles/Dockerfile.rpm" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg BASE_IMAGE_VERSION="$BASE_IMAGE_VERSION" \
    -t "$IMAGE_TAG" "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
CID=$(docker create "$IMAGE_TAG")
docker cp "$CID:/output/." "$OUTPUT_DIR/"
docker rm "$CID" > /dev/null

echo ""; echo ">>> Output:"; ls -lh "$OUTPUT_DIR"
