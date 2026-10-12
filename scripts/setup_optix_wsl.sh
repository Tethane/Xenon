#!/usr/bin/env bash
# Extract the matching userspace runtime; never installs or changes a driver.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime_dir=${1:-"$repo_dir/build/optix-runtime"}
cache_dir=${XDG_CACHE_HOME:-"$HOME/.cache"}/xenon
version=590.48.01
archive="$cache_dir/NVIDIA-Linux-x86_64-$version.run"
extract_dir="$cache_dir/nvidia-$version"
mkdir -p "$cache_dir" "$runtime_dir"
if [[ ! -f "$archive" ]]; then
	curl --fail --location --retry 3 \
		"https://download.nvidia.com/XFree86/Linux-x86_64/$version/NVIDIA-Linux-x86_64-$version.run" \
		--output "$archive"
fi
printf '%s  %s\n' b9e2f80693781431cc87f4cd29109e133dcecb50a50d6b68d4b3bf2d696bd689 "$archive" | sha256sum --check
if [[ ! -f "$extract_dir/libnvoptix.so.$version" ]]; then
	sh "$archive" --extract-only --target "$extract_dir"
fi
for library in libnvoptix libnvidia-rtcore libnvidia-gpucomp; do
	cp "$extract_dir/$library.so.$version" "$runtime_dir/"
done
cp "$extract_dir/nvoptix.bin" "$runtime_dir/"
ln -sfn "libnvoptix.so.$version" "$runtime_dir/libnvoptix.so.1"
ln -sfn "libnvidia-gpucomp.so.$version" "$runtime_dir/libnvidia-gpucomp.so"
printf '\nLocal runtime prepared. Run Xenon using:\nLD_LIBRARY_PATH=%q:/usr/lib/wsl/lib build/optix/xenon --backend optix ...\n' "$runtime_dir"
