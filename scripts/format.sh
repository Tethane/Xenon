#!/usr/bin/env bash
# Format first-party source; keep dependencies and generated lookup data intact.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-}"
if [[ -n "$mode" && "$mode" != --check ]]; then
	echo "Usage: scripts/format.sh [--check]" >&2
	exit 2
fi
for formatter in clang-format ruff shfmt cmake-format; do
	command -v "$formatter" >/dev/null || {
		echo "Missing $formatter; see docs/DEVELOPMENT.md" >&2
		exit 1
	}
done
mapfile -d '' cpp < <(rg --files -0 src tests scripts -g '*.cpp' -g '*.h' -g '*.cu' -g '*.cuh')
mapfile -d '' python < <(rg --files -0 scripts tests scenes -g '*.py')
python+=(.cmake-format.py)
mapfile -d '' shell < <(rg --files -0 scripts -g '*.sh')
shell+=(xenon.sh)
mapfile -d '' cmake < <(rg --files -0 src scripts tests -g '*.cmake')
cmake+=(CMakeLists.txt)
if [[ "$mode" == --check ]]; then
	clang-format --dry-run --Werror "${cpp[@]}"
	ruff format --check "${python[@]}"
	shfmt -d "${shell[@]}"
	cmake-format --check "${cmake[@]}"
else
	clang-format -i "${cpp[@]}"
	ruff format "${python[@]}"
	shfmt -w "${shell[@]}"
	cmake-format -i "${cmake[@]}"
fi
