#!/usr/bin/env python3
"""Compare traversal and scheduling controls using the same Xenon executable.

Runs variants in round-robin order, excludes one warmup per variant and compares
untonemapped RGB images. Requires NumPy, like benchmark.py. Preserve the runtime
LD_LIBRARY_PATH when OptiX needs an explicitly selected driver library.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from benchmark import (
	compare,
	read_pfm,
	command_output,
	file_digest,
	source_snapshot,
	asset_snapshot,
	build_snapshot,
	validate_metrics,
)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("scene", type=Path)
	parser.add_argument("--binary", type=Path, default=Path("build/optix/xenon"))
	parser.add_argument("--output-dir", type=Path, default=Path("build/scheduling-benchmark"))
	parser.add_argument("--samples", type=int, default=128)
	parser.add_argument("--width", type=int, default=512)
	parser.add_argument("--height", type=int, default=384)
	parser.add_argument("--repeats", type=int, default=3)
	args = parser.parse_args()
	if min(args.samples, args.width, args.height, args.repeats) < 1:
		parser.error("samples, dimensions and repeats must be positive")
	binary, scene, destination = (
		args.binary.resolve(),
		args.scene.resolve(),
		args.output_dir.resolve(),
	)
	destination.mkdir(parents=True, exist_ok=True)
	variants = [
		("cuda-host", "cuda", {"XENON_GPU_HOST_COUNTERS": "1"}),
		("optix-host", "optix", {"XENON_GPU_HOST_COUNTERS": "1", "XENON_OPTIX_UPLOAD_PARAMS": "1"}),
		("optix-cached", "optix", {"XENON_GPU_HOST_COUNTERS": "1"}),
		("cuda-device", "cuda", {}),
		("optix-device", "optix", {}),
	]
	provenance = {
		"binary_sha256": file_digest(binary),
		"asset_snapshot": asset_snapshot(scene),
		"source_snapshot": source_snapshot(),
		"build_snapshot": build_snapshot(binary),
	}
	rows = []
	reference = None
	for repeat in range(args.repeats + 1):
		order = variants[repeat % len(variants) :] + variants[: repeat % len(variants)]
		for name, backend, controls in order:
			target = destination / f"{name}-{repeat}"
			env = dict(os.environ)
			for key in ("XENON_GPU_HOST_COUNTERS", "XENON_OPTIX_UPLOAD_PARAMS"):
				env.pop(key, None)
			env.update(controls)
			command = [
				str(binary),
				"--scene",
				str(scene),
				"--backend",
				backend,
				"--width",
				str(args.width),
				"--height",
				str(args.height),
				"--samples",
				str(args.samples),
				"--min-bounces",
				"3",
				"--max-bounces",
				"8",
				"--seed",
				"42",
				"--no-display",
				"--output",
				str(target) + ".png",
				"--linear-output",
				str(target) + ".pfm",
				"--metrics",
				str(target) + ".json",
			]
			with open(str(target) + ".log", "w") as log:
				subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
			metrics = json.loads(Path(str(target) + ".json").read_text())
			validate_metrics(metrics, backend, args.samples, args.width, args.height, 3, 8, 42, 0)
			linear = read_pfm(str(target) + ".pfm")
			if reference is None:
				reference = linear
			rows.append(
				{
					"variant": name,
					"run": repeat,
					"warmup": repeat == 0,
					"controls": controls,
					"metrics": metrics,
					"command": command,
					"linear_sha256": file_digest(str(target) + ".pfm"),
					"linear_vs_cuda_host": compare(linear, reference),
				}
			)
			print(f"{name} run {repeat}: {metrics['seconds']['render']:.6f} s", flush=True)
	summary = {}
	for name, _, _ in variants:
		measured = [row for row in rows if row["variant"] == name and not row["warmup"]]
		times = [row["metrics"]["seconds"]["render"] for row in measured]
		totals = [row["metrics"]["seconds"]["total"] for row in measured]
		summary[name] = {
			"render_median": statistics.median(times),
			"render_min": min(times),
			"render_max": max(times),
			"total_median": statistics.median(totals),
			"worst_relative_rmse": max(
				row["linear_vs_cuda_host"]["relative_rmse"] for row in measured
			),
		}
	after_binary = file_digest(binary)
	after_assets = asset_snapshot(scene)
	report = {
		"schema_version": 1,
		"binary": str(binary),
		"binary_sha256": provenance["binary_sha256"],
		"provenance": provenance,
		"provenance_recheck": {
			"binary_sha256": after_binary,
			"asset_sha256": after_assets["sha256"],
			"stable": after_binary == provenance["binary_sha256"]
			and after_assets["sha256"] == provenance["asset_snapshot"]["sha256"],
		},
		"scene": str(scene),
		"scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
		"nvidia_smi": command_output(
			["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
		),
		"rows": rows,
		"summary": summary,
	}
	(destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
	if not report["provenance_recheck"]["stable"]:
		raise RuntimeError(
			"Executable or referenced assets changed during scheduling benchmark; timings are invalid"
		)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
