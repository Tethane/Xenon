#!/usr/bin/env bash
# xenon.sh — build and run the Xenon path tracer
# Usage: ./xenon.sh [--build] [--debug] [--samples N] [--scene FILE]
#                   [--output FILE] [--no-display] [--benchmark]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="Release"
FORCE_BUILD=0
SAMPLES=256
TEST_SCENE="cornell_box"
SCENE=""
OUTPUT=""
NO_DISPLAY=""
BENCHMARK=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)         BUILD_TYPE="Debug"; shift ;;
    --build)         FORCE_BUILD=1; shift ;;
    --samples)       SAMPLES="$2"; shift 2 ;;
    --bounces)       BOUNCES="$2"; shift 2 ;;
    --outdir)        OUTDIR="$2"; shift 2 ;;
    --scene)         SCENE="$2"; shift 2 ;;
    --test-scene)    TEST_SCENE="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --render-mode)   RENDER_MODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

BOUNCES=${BOUNCES:-8}
OUTDIR=${OUTDIR:-renders}

if [[ -z "${SCENE}" ]]; then
  SCENE="${ROOT_DIR}/scenes/${TEST_SCENE}/${TEST_SCENE}.xenon"
fi
OUTPUT=${OUTPUT:-"${TEST_SCENE}_output.png"}

mkdir -p "${ROOT_DIR}/${OUTDIR}"

BUILD_DIR="${ROOT_DIR}/build/$(echo "${BUILD_TYPE}" | tr '[:upper:]' '[:lower:]')"

# ── Build ─────────────────────────────────────────────────────────────────────
needs_build() {
  [[ $FORCE_BUILD -eq 1 ]] && return 0
  [[ ! -f "${BUILD_DIR}/xenon" ]] && return 0
  # Rebuild if any source file is newer than the binary
  if find "${ROOT_DIR}/src" "${ROOT_DIR}/CMakeLists.txt" \
       -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" 2>/dev/null \
     | xargs -I{} find {} -newer "${BUILD_DIR}/xenon" 2>/dev/null \
     | grep -q .; then
    return 0
  fi
  return 1
}

if needs_build; then
  echo "==> Configuring (${BUILD_TYPE})..."
  cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -Wno-dev 2>&1

  echo "==> Building..."
  cmake --build "${BUILD_DIR}" --parallel "$(nproc)" 2>&1
  echo "==> Build complete."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
exec "${BUILD_DIR}/xenon" \
  --samples "${SAMPLES}" \
  --scene "${SCENE}" \
  --output "${OUTPUT}" \
  --output-dir "${OUTDIR}" \
  --max-bounces "${BOUNCES}" \
  --render-mode "${RENDER_MODE:-0}"
