#!/usr/bin/env bash
# Build the renderer incrementally and run a named scene or .xenon file.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_TYPE="Release"
FORCE_BUILD=0
LIST_SCENES=0
SCENE="cornell_box"
BACKEND="cpu"
ARGS=()

help() {
	cat <<'EOF'
Usage: ./xenon.sh [--scene NAME|FILE.xenon] [--backend cpu|cuda|optix] [OPTIONS]

Builds xenon incrementally, then renders. Default: cornell_box, CPU.
  ./xenon.sh --scene prim_types --backend cpu --samples 64
  ./xenon.sh --scene lucy --backend optix --samples 512
  ./xenon.sh --scene sponza --backend optix --samples 1024

  --help, -h             Show help without building.
  --list-scenes          List bundled names and benchmark aliases; do not build.
  --scene NAME|FILE      Bundled name, benchmark alias, or .xenon file path.
  --debug                Use Debug instead of Release.
  --build                Reconfigure CMake before building and running.
  --backend cpu|cuda|optix   Requested backend; never silently falls back.
  --samples N --width W --height H
  --min-bounces N --max-bounces N   (--bounces aliases --max-bounces).
  --seed N --exposure EV --no-display
  --output FILE          PNG filename (default: <scene>_output.png).
  --output-dir DIR       Output directory (default: renders; alias: --outdir).
  --linear-output FILE.pfm --metrics FILE.json

sponza / classroom and manifest IDs use prepared benchmark scenes and their
resolution, exposure, bounce and seed defaults. Explicit flags override defaults.
Assets are not downloaded. See docs/BLENDER_IMPORT.md for scene preparation.

Environment:
  XENON_BUILD_DIR        Build directory, relative to repo root if not absolute.
                        Default: build/release (build/debug with --debug).
  XENON_JOBS             Positive build parallelism; defaults to at most 4 jobs.
                        CMAKE_BUILD_PARALLEL_LEVEL is also honored.
  CUDACXX / PATH         CUDA compiler for initial GPU configuration.
                        Existing CMake compiler/toolkit settings are preserved.

A fresh CPU build needs no CUDA/OptiX SDK. GPU requests enable the needed backend.
On WSL, an existing build/optix-runtime is added to the loader path for OptiX.
See docs/GETTING_STARTED.md for dependencies and setup.
EOF
}
fail() {
	echo "[Error] $*" >&2
	exit 1
}

while [ $# -gt 0 ]; do
	case "$1" in
	--help | -h)
		help
		exit 0
		;;
	--list-scenes)
		LIST_SCENES=1
		shift
		;;
	--debug)
		BUILD_TYPE="Debug"
		shift
		;;
	--build)
		FORCE_BUILD=1
		shift
		;;
	--no-display)
		ARGS+=("$1")
		shift
		;;
	--scene | --samples | --output | --output-dir | --outdir | --bounces | --min-bounces | --max-bounces | --backend | --exposure | --width | --height | --seed | --linear-output | --metrics)
		[[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
		case "$1" in
		--scene) SCENE="$2" ;;
		--backend) BACKEND="$2" ;;
		--bounces) ARGS+=(--max-bounces "$2") ;;
		--outdir) ARGS+=(--output-dir "$2") ;;
		*) ARGS+=("$1" "$2") ;;
		esac
		shift 2
		;;
	*) fail "Unknown argument: $1 (see --help)" ;;
	esac
done

case "$BACKEND" in cpu | cuda | optix) ;; *) fail "Unknown backend: $BACKEND (expected cpu, cuda or optix)" ;; esac

# Python is already a CMake prerequisite. Read benchmark defaults from the same
# manifest as qualification; keep resolved paths in individual array entries.
SCENE_INFO="$(
	python3 - "$ROOT_DIR" "$SCENE" "$LIST_SCENES" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

root, name, listing = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
manifest = json.loads((root / "scenes/benchmarks/suite.json").read_text())
entries = {entry["id"]: entry for entry in manifest["scenes"]}
aliases = {"sponza": "sponza-full", "classroom": "classroom-instances"}
paths = sorted((root / "scenes").glob("*/*.xenon"))
counts = Counter(path.stem for path in paths)
bundled = {
	path.stem: path for path in paths if counts[path.stem] == 1 or path.stem == path.parent.name
}
if listing:
	print("Bundled scene names (external meshes may require asset setup):")
	for key, path in bundled.items():
		print(f"  {key:24} {path.relative_to(root)}")
	print("\nBenchmark aliases (prepared using docs/BLENDER_IMPORT.md):")
	for key, entry in entries.items():
		label = " / ".join([alias for alias, target in aliases.items() if target == key] + [key])
		state = "scene file present" if (root / entry["scene"]).is_file() else "needs preparation"
		print(
			f"  {label:32} {entry['width']}x{entry['height']}, {entry.get('exposure', 0):+g} EV; {state}"
		)
	print("\nA .xenon file path can also be passed to --scene.")
	sys.exit(0)
defaults = []
if Path(name).is_file():
	path = Path(name).resolve()
	output_name = path.stem
elif name in bundled:
	path = bundled[name]
	output_name = name
elif aliases.get(name, name) in entries:
	entry = entries[aliases.get(name, name)]
	path = root / entry["scene"]
	output_name = name
	if not path.is_file():
		sys.exit(
			f"[Error] Benchmark scene '{name}' is not prepared: {path}\n"
			"Follow docs/BLENDER_IMPORT.md#reproduce-the-final-five-scene-suite; no assets were downloaded."
		)
	values = {**manifest.get("defaults", {}), **entry}
	for key in ("width", "height", "exposure", "min_bounces", "max_bounces", "seed"):
		if key in values:
			defaults.extend(["--" + key.replace("_", "-"), str(values[key])])
else:
	sys.exit(
		f"[Error] Could not find scene: {name}\nUse --list-scenes or pass an existing .xenon file."
	)
if "\n" in str(path) or "\n" in output_name:
	sys.exit("[Error] Scene paths containing newlines are not supported by the launcher.")
print(path)
print(output_name + "_output.png")
print("\n".join(defaults), end="")
PY
)" || exit 1
if [[ $LIST_SCENES -eq 1 ]]; then
	printf '%s\n' "$SCENE_INFO"
	exit 0
fi
mapfile -t SCENE_FIELDS <<<"$SCENE_INFO"
SCENE_PATH="${SCENE_FIELDS[0]}"
OUTPUT_NAME="${SCENE_FIELDS[1]}"
SCENE_DEFAULTS=("${SCENE_FIELDS[@]:2}")

BUILD_DIR="${XENON_BUILD_DIR:-build/${BUILD_TYPE,,}}"
[[ "$BUILD_DIR" = /* ]] || BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
JOBS="${XENON_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
if [[ -z "$JOBS" ]]; then
	JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
	((JOBS <= 4)) || JOBS=4
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "XENON_JOBS / CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"

cache_value() {
	[[ -f "${BUILD_DIR}/CMakeCache.txt" ]] || return 0
	local key value
	while IFS='=' read -r key value; do
		if [[ "$key" == "$1":* ]]; then
			printf '%s' "$value"
			return 0
		fi
	done <"${BUILD_DIR}/CMakeCache.txt"
}
CONFIGURE=$FORCE_BUILD
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE="$BUILD_TYPE")
if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
	CONFIGURE=1
	CMAKE_ARGS+=(-DXENON_ENABLE_CUDA=OFF -DXENON_ENABLE_OPTIX=OFF)
elif [[ "$(cache_value CMAKE_BUILD_TYPE)" != "$BUILD_TYPE" ]]; then
	CONFIGURE=1
fi
if [[ "$BACKEND" != cpu ]]; then
	CMAKE_ARGS+=(-DXENON_ENABLE_CUDA=ON)
	if [[ "$(cache_value XENON_ENABLE_CUDA)" != ON || -z "$(cache_value CMAKE_CUDA_COMPILER)" || "$(cache_value CMAKE_CUDA_COMPILER)" == *NOTFOUND ]]; then CONFIGURE=1; fi
	if [[ "$(cache_value CMAKE_CUDA_COMPILER)" == *NOTFOUND ]]; then CMAKE_ARGS+=(-UCMAKE_CUDA_COMPILER); fi
	if [[ "$BACKEND" == optix ]]; then
		CMAKE_ARGS+=(-DXENON_ENABLE_OPTIX=ON)
		[[ "$(cache_value XENON_ENABLE_OPTIX)" == ON ]] || CONFIGURE=1
	elif [[ -z "$(cache_value XENON_ENABLE_OPTIX)" ]]; then
		CMAKE_ARGS+=(-DXENON_ENABLE_OPTIX=OFF)
	fi
fi
if [[ $CONFIGURE -eq 1 ]]; then
	echo "[Launcher] Configuring $BUILD_TYPE in $BUILD_DIR"
	if ! cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"; then
		[[ "$BACKEND" == cpu ]] || echo "[Error] GPU configuration failed. Install CUDA and set CUDACXX or put nvcc on PATH; see docs/GETTING_STARTED.md." >&2
		exit 1
	fi
fi
if [[ "$BACKEND" != cpu ]]; then
	[[ -n "$(cache_value CMAKE_CUDA_COMPILER)" && "$(cache_value CMAKE_CUDA_COMPILER)" != *NOTFOUND ]] || fail "Requested $BACKEND, but CUDA was not configured. Set CUDACXX or put nvcc on PATH; no CPU fallback was selected."
fi
echo "[Launcher] Building xenon ($BUILD_TYPE, $JOBS jobs)"
cmake --build "$BUILD_DIR" --target xenon --config "$BUILD_TYPE" --parallel "$JOBS"
BINARY="${BUILD_DIR}/xenon"
if [[ -n "$(cache_value CMAKE_CONFIGURATION_TYPES)" ]]; then BINARY="${BUILD_DIR}/${BUILD_TYPE}/xenon"; fi
[[ -x "$BINARY" ]] || fail "Build did not produce the renderer: $BINARY"
if [[ "$BACKEND" == optix && "$(uname -r)" == *[Mm]icrosoft* && -f "${ROOT_DIR}/build/optix-runtime/libnvoptix.so.1" ]]; then
	export LD_LIBRARY_PATH="${ROOT_DIR}/build/optix-runtime:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
	echo "[Launcher] Using the local WSL OptiX runtime in build/optix-runtime"
fi
echo "[Launcher] Rendering $SCENE with $BACKEND"
exec "$BINARY" --scene "$SCENE_PATH" --backend "$BACKEND" \
	--output "$OUTPUT_NAME" "${SCENE_DEFAULTS[@]}" "${ARGS[@]}"
