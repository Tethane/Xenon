#!/usr/bin/env python3
"""Run matched linear-HDR renders and report measured error and timing spread.

Requires NumPy. All subprocess failures are fatal; a requested backend never falls
back. A reference image is an independent measurement, not inferred from blur.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shlex
import subprocess
import time
from datetime import datetime, timezone

import numpy as np


def read_pfm(path):
	with Path(path).open("rb") as stream:
		if stream.readline().strip() != b"PF":
			raise ValueError(f"Expected RGB PFM: {path}")
		line = stream.readline()
		while line.startswith(b"#"):
			line = stream.readline()
		width, height = map(int, line.split())
		scale = float(stream.readline())
		if width <= 0 or height <= 0 or scale == 0:
			raise ValueError("Invalid PFM header")
		data = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
		if data.size != width * height * 3:
			raise ValueError("Invalid PFM payload length")
		image = data.reshape(height, width, 3)[::-1].astype(np.float64) * abs(scale)
		if not np.isfinite(image).all():
			raise ValueError(f"Nonfinite reference/image: {path}")
		return image


def compare(image, reference):
	if image.shape != reference.shape:
		raise ValueError(f"Image shapes differ: {image.shape} vs {reference.shape}")
	difference = image - reference
	mse = float(np.mean(difference**2))
	ref_rms = float(np.sqrt(np.mean(reference**2)))
	# HDR has no fixed peak intensity; PSNR therefore requires an explicit peak
	# and is intentionally not reported here.
	weights = np.array([0.2126, 0.7152, 0.0722])
	luma = image @ weights
	ref_luma = reference @ weights
	return {
		"rmse": float(np.sqrt(mse)),
		"relative_rmse": float(np.sqrt(mse) / max(ref_rms, 1e-12)),
		"mae": float(np.mean(np.abs(difference))),
		"max_absolute_error": float(np.max(np.abs(difference))),
		"mean_luminance": float(np.mean(luma)),
		"reference_mean_luminance": float(np.mean(ref_luma)),
		"relative_luminance_bias": float(
			(np.mean(luma) - np.mean(ref_luma)) / max(abs(np.mean(ref_luma)), 1e-12)
		),
	}


def command_output(command):
	try:
		return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def source_snapshot():
	"""Include new source files as well as tracked modifications in provenance."""
	names = command_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
	if names is None:
		return None
	files = {}
	for name in sorted(set(names.splitlines())):
		path = Path(name)
		if path.is_file() and (
			path.suffix in {".h", ".cuh", ".cpp", ".cu", ".inc", ".cmake", ".py", ".sh"}
			or name == "CMakeLists.txt"
		):
			files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
	return {
		"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
		"files": files,
	}


def file_digest(path):
	digest = hashlib.sha256()
	with Path(path).open("rb") as stream:
		while block := stream.read(1024 * 1024):
			digest.update(block)
	return digest.hexdigest()


def asset_snapshot(scene):
	"""Fingerprint scene/material files and the exact referenced mesh/map bytes."""
	files = {}

	def visit(path):
		path = path.resolve()
		if str(path) in files:
			return
		files[str(path)] = file_digest(path)
		if path.suffix not in {".xenon", ".mat"}:
			return
		commands = [shlex.split(line, comments=True) for line in path.read_text().splitlines()]
		version2 = any(words == ["version", "2"] for words in commands)
		for words in commands:
			if len(words) < 2:
				continue
			key = words[0]
			if key in {"mesh", "environment", "matfile"} or key.endswith("Texture"):
				base = (
					path.parent / "materials" if key == "matfile" and not version2 else path.parent
				)
				visit(base / words[1])

	visit(scene)
	return {
		"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
		"files": files,
	}


def build_snapshot(binary):
	"""Record available build files and explicitly selected runtime candidates.

	These files describe the adjacent build directory. Their hashes do not by
	themselves prove which sources produced a copied executable.
	"""
	root = Path(binary).resolve().parent
	cache = root / "CMakeCache.txt"
	values = {}
	if cache.exists():
		for line in cache.read_text().splitlines():
			if not line or line.startswith(("#", "//")) or "=" not in line:
				continue
			key, value = line.split("=", 1)
			values[key.split(":", 1)[0]] = value
	selected = {
		key: value
		for key, value in values.items()
		if key
		in {
			"CMAKE_BUILD_TYPE",
			"CMAKE_CXX_COMPILER",
			"CMAKE_CUDA_COMPILER",
			"CMAKE_CUDA_ARCHITECTURES",
			"CMAKE_CXX_STANDARD",
			"CMAKE_CUDA_STANDARD",
			"CMAKE_CUDA_HOST_COMPILER",
			"OPTIX_INCLUDE_DIR",
			"XENON_ENABLE_CUDA",
			"XENON_ENABLE_OPTIX",
			"XENON_TEST_CUDA",
		}
		or (
			key.startswith(("CMAKE_CXX_FLAGS", "CMAKE_CUDA_FLAGS", "CMAKE_EXE_LINKER_FLAGS"))
			and not key.endswith("-ADVANCED")
		)
	}
	files = {
		str(path): file_digest(path)
		for path in [cache, root / "compile_commands.json", root / "optix_programs.ptx"]
		if path.is_file()
	}
	compiler = values.get("CMAKE_CUDA_COMPILER")
	candidates = []
	seen = set()
	# Record candidates found in the caller's explicit loader path. Do not
	# claim these are the actually loaded libraries, which needs process maps.
	for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
		if not directory:
			continue
		for name in ("libnvoptix.so.1", "libnvidia-rtcore.so.1", "libnvidia-gpucomp.so"):
			path = Path(directory) / name
			if path.is_file() and path.resolve() not in seen:
				resolved = path.resolve()
				seen.add(resolved)
				candidates.append(
					{
						"requested": str(path),
						"resolved": str(resolved),
						"sha256": file_digest(resolved),
					}
				)
	return {
		"directory": str(root),
		"cmake_cache": selected,
		"file_sha256": files,
		"nvcc_version": command_output([compiler, "--version"])
		if compiler and Path(compiler).is_file()
		else None,
		"explicit_runtime_candidates": candidates,
		"runtime_note": "Candidates from LD_LIBRARY_PATH in search order; actual process-loaded paths were not inspected.",
	}


def validate_metrics(
	metrics, backend, samples, width, height, min_bounces, max_bounces, seed, exposure
):
	expected = {
		"backend": backend,
		"samples": samples,
		"width": width,
		"height": height,
		"min_bounces": min_bounces,
		"max_bounces": max_bounces,
		"seed": seed,
		"exposure_ev": exposure,
	}

	def matches(key, value):
		actual = metrics.get(key)
		if key == "exposure_ev" and type(actual) in (int, float):
			# Xenon parses exposure as float32 and prints 12 significant digits.
			# Compare the represented float32 values, accepting that round trip
			# without accepting even a one-ULP change in the renderer setting.
			return (
				math.isfinite(actual)
				and math.isfinite(value)
				and np.float32(actual) == np.float32(value)
			)
		return actual == value

	mismatched = {
		key: {"requested": value, "reported": metrics.get(key)}
		for key, value in expected.items()
		if not matches(key, value)
	}
	if mismatched:
		raise RuntimeError(f"Renderer changed requested settings: {mismatched}")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("scene", type=Path)
	parser.add_argument("--binary", type=Path, default=Path("build/release/xenon"))
	parser.add_argument(
		"--backends", nargs="+", choices=["cpu", "cuda", "optix"], default=["cpu", "cuda", "optix"]
	)
	parser.add_argument("--samples", type=int, default=64)
	parser.add_argument(
		"--sample-levels",
		type=int,
		nargs="+",
		help="Measure a quality/time curve at these sample counts",
	)
	parser.add_argument("--repeats", type=int, default=3)
	parser.add_argument("--warmups", type=int, default=1)
	parser.add_argument("--width", type=int, default=320)
	parser.add_argument("--height", type=int, default=240)
	parser.add_argument("--min-bounces", type=int, default=3)
	parser.add_argument("--max-bounces", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--exposure", type=float, default=0)
	parser.add_argument(
		"--reference", type=Path, help="Independent high-sample scene-linear RGB PFM"
	)
	parser.add_argument(
		"--target-error", type=float, help="Report first tested sample level meeting relative RMSE"
	)
	parser.add_argument("--output-dir", type=Path, default=Path("build/qualification"))
	args = parser.parse_args()
	levels = sorted(set(args.sample_levels or [args.samples]))
	if min(levels) <= 0 or args.repeats <= 0 or args.warmups < 0:
		parser.error("samples/repeats must be positive and warmups nonnegative")
	if args.target_error is not None and (args.reference is None or args.target_error <= 0):
		parser.error("positive --target-error requires --reference")
	binary, scene = args.binary.resolve(), args.scene.resolve()
	destination = args.output_dir.resolve()
	destination.mkdir(parents=True, exist_ok=True)
	reference = read_pfm(args.reference) if args.reference else None
	if reference is not None and reference.shape != (args.height, args.width, 3):
		parser.error("Reference resolution must match --width and --height")
	report = {
		"schema_version": 1,
		"started_utc": datetime.now(timezone.utc).isoformat(),
		"build_snapshot": build_snapshot(binary),
		"scene": str(scene),
		"scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
		"binary": str(binary),
		"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
		"git_commit": command_output(["git", "rev-parse", "HEAD"]),
		"git_diff_sha256": hashlib.sha256(
			(command_output(["git", "diff"]) or "").encode()
		).hexdigest(),
		"source_snapshot": source_snapshot(),
		"asset_snapshot": asset_snapshot(scene),
		"nvidia_smi": command_output(
			[
				"nvidia-smi",
				"--query-gpu=name,driver_version,memory.total,power.limit",
				"--format=csv,noheader",
			]
		),
		"reference": str(args.reference.resolve()) if args.reference else None,
		"reference_sha256": file_digest(args.reference) if args.reference else None,
		"settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
		"runs": [],
		"summary": {},
		"comparisons": {},
		"notes": [
			"Repeated runs use the same seed to measure timing spread at identical requested quality.",
			"Linear RMSE includes sampling noise; CPU agreement cannot rule out a shared transport error.",
			"seconds.total ends after image output, before metrics output and process teardown; wall_seconds includes the whole subprocess.",
			"GPU free memory deltas include HDR allocations and context/allocator effects; they are approximate. Console scene-upload MB currently omits HDR buffers.",
			"Time to target is the first tested sample level meeting the threshold, with no interpolation.",
			"Adjacent build files and source hashes are provenance records, not proof of executable compilation lineage.",
			"Binary and all referenced asset hashes are rechecked after the measured runs; changes invalidate the report.",
		],
	}
	# Interleave backends per repetition to reduce sustained-heat/order effects.
	for samples in levels:
		for repetition in range(-args.warmups, args.repeats):
			order = (
				args.backends[repetition % len(args.backends) :]
				+ args.backends[: repetition % len(args.backends)]
			)
			for backend in order:
				stem = f"{backend}-{samples}spp-{'warmup' + str(-repetition) if repetition < 0 else 'run' + str(repetition + 1)}"
				base = destination / stem
				command = [
					str(binary),
					"--scene",
					str(scene),
					"--backend",
					backend,
					"--no-display",
					"--samples",
					str(samples),
					"--width",
					str(args.width),
					"--height",
					str(args.height),
					"--min-bounces",
					str(args.min_bounces),
					"--max-bounces",
					str(args.max_bounces),
					"--seed",
					str(args.seed),
					"--exposure",
					str(args.exposure),
					"--output",
					str(base.with_suffix(".png")),
					"--linear-output",
					str(base.with_suffix(".pfm")),
					"--metrics",
					str(base.with_suffix(".json")),
				]
				print(f"Rendering {stem}", flush=True)
				start = time.monotonic()
				with base.with_suffix(".log").open("w") as log:
					result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
				wall_seconds = time.monotonic() - start
				if result.returncode:
					raise RuntimeError(
						f"{backend} failed ({result.returncode}); see {base.with_suffix('.log')}"
					)
				metrics = json.loads(base.with_suffix(".json").read_text())
				validate_metrics(
					metrics,
					backend,
					samples,
					args.width,
					args.height,
					args.min_bounces,
					args.max_bounces,
					args.seed,
					args.exposure,
				)
				metrics["wall_seconds"] = wall_seconds
				metrics["linear_sha256"] = file_digest(base.with_suffix(".pfm"))
				metrics["command"] = command
				metrics["warmup"] = repetition < 0
				metrics["repetition"] = repetition + 1
				if reference is not None:
					metrics["reference_error"] = compare(
						read_pfm(base.with_suffix(".pfm")), reference
					)
				report["runs"].append(metrics)
				(destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
		for backend in args.backends:
			runs = [
				r
				for r in report["runs"]
				if r["backend"] == backend and r["samples"] == samples and not r["warmup"]
			]
			key = f"{backend}-{samples}spp"
			summary = {}
			for stage in [*runs[0]["seconds"], "wall"]:
				timings = [
					r["wall_seconds"] if stage == "wall" else r["seconds"][stage] for r in runs
				]
				summary[stage] = {
					"median": statistics.median(timings),
					"min": min(timings),
					"max": max(timings),
					"stdev": statistics.stdev(timings) if len(timings) > 1 else 0,
				}
			if reference is not None:
				summary["reference_error"] = runs[0]["reference_error"]
			report["summary"][key] = summary
		first = args.backends[0]
		reference_run = next(
			r
			for r in report["runs"]
			if r["backend"] == first and r["samples"] == samples and not r["warmup"]
		)
		first_image = read_pfm(reference_run["linear_pfm"])
		for backend in args.backends[1:]:
			run = next(
				r
				for r in report["runs"]
				if r["backend"] == backend and r["samples"] == samples and not r["warmup"]
			)
			report["comparisons"][f"{backend}-vs-{first}-{samples}spp"] = compare(
				read_pfm(run["linear_pfm"]), first_image
			)
		if "cuda" in args.backends and "optix" in args.backends:
			cuda, optix = (report["summary"][f"{b}-{samples}spp"] for b in ["cuda", "optix"])
			report.setdefault("optix_speedup", {})[str(samples)] = {
				"render": cuda["render"]["median"] / optix["render"]["median"],
				"through_image_output": cuda["total"]["median"] / optix["total"]["median"],
				"process_wall": cuda["wall"]["median"] / optix["wall"]["median"],
			}
	if args.target_error is not None:
		report["time_to_target"] = {}
		for backend in args.backends:
			passing = [
				(n, report["summary"][f"{backend}-{n}spp"])
				for n in levels
				if report["summary"][f"{backend}-{n}spp"]["reference_error"]["relative_rmse"]
				<= args.target_error
			]
			report["time_to_target"][backend] = (
				None
				if not passing
				else {
					"samples": passing[0][0],
					"render_seconds": passing[0][1]["render"]["median"],
					"total_seconds": passing[0][1]["total"]["median"],
				}
			)
	after_binary = file_digest(binary)
	after_assets = asset_snapshot(scene)
	report["finished_utc"] = datetime.now(timezone.utc).isoformat()
	report["provenance_recheck"] = {
		"binary_sha256": after_binary,
		"asset_sha256": after_assets["sha256"],
		"stable": after_binary == report["binary_sha256"]
		and after_assets["sha256"] == report["asset_snapshot"]["sha256"],
	}
	(destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
	if not report["provenance_recheck"]["stable"]:
		raise RuntimeError(
			"Executable or referenced assets changed during benchmarking; recorded timings are invalid"
		)
	rows = [
		"backend,samples,render_median_s,render_min_s,render_max_s,total_median_s,relative_rmse"
	]
	for backend in args.backends:
		for samples in levels:
			row = report["summary"][f"{backend}-{samples}spp"]
			rows.append(
				f"{backend},{samples},{row['render']['median']:.6f},{row['render']['min']:.6f},"
				f"{row['render']['max']:.6f},{row['total']['median']:.6f},"
				f"{row.get('reference_error', {}).get('relative_rmse', '')}"
			)
	(destination / "timings.csv").write_text("\n".join(rows) + "\n")
	print("\n".join(rows))
	print(f"Results: {destination / 'results.json'}")


if __name__ == "__main__":
	main()
