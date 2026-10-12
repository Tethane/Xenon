#!/usr/bin/env python3
"""Run the checked-in real-scene qualification manifest through benchmark.py.

Assets and matched scene-linear references must already exist. This command
never downloads assets, converts scenes, or substitutes a different reference.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from benchmark import file_digest, read_pfm

REPO = Path(__file__).resolve().parents[1]


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--binary", type=Path, default=REPO / "build/optix/xenon")
	parser.add_argument("--manifest", type=Path, default=REPO / "scenes/benchmarks/suite.json")
	parser.add_argument("--scenes", nargs="+", help="Manifest scene IDs; defaults to every scene")
	parser.add_argument("--samples", type=int, default=128)
	parser.add_argument("--repeats", type=int, default=3)
	parser.add_argument("--warmups", type=int, default=1)
	parser.add_argument(
		"--backends", nargs="+", choices=["cpu", "cuda", "optix"], default=["cpu", "cuda", "optix"]
	)
	parser.add_argument("--output-dir", type=Path, default=REPO / "build/qualification/final")
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Validate all inputs and print commands without rendering",
	)
	args = parser.parse_args()
	if args.samples < 1 or args.repeats < 1 or args.warmups < 0:
		parser.error("samples/repeats must be positive and warmups nonnegative")
	manifest_path = args.manifest.resolve()
	manifest = json.loads(manifest_path.read_text())
	if manifest.get("schema_version") != 1:
		parser.error("unsupported qualification manifest schema")
	entries = {entry["id"]: entry for entry in manifest["scenes"]}
	if len(entries) != len(manifest["scenes"]):
		parser.error("manifest has duplicate scene IDs")
	selected = args.scenes or list(entries)
	unknown = set(selected) - entries.keys()
	if unknown:
		parser.error(f"unknown scene IDs: {sorted(unknown)}; available: {list(entries)}")
	if len(set(selected)) != len(selected) or len(set(args.backends)) != len(args.backends):
		parser.error("scene and backend selections must be unique")
	binary = args.binary.resolve()
	destination = args.output_dir.resolve()
	if not binary.is_file():
		parser.error(f"renderer does not exist: {binary}")
	defaults = manifest["defaults"]
	commands = {}
	references = {}
	# Preflight every requested scene before starting any expensive render.
	for name in selected:
		entry = entries[name]
		paths = {
			key: (REPO / entry[key]).resolve()
			for key in ("scene", "reference", "reference_metrics")
		}
		for key, path in paths.items():
			if not path.is_file():
				parser.error(f"{name}: missing {key}: {path}; generate the documented assets first")
		shape = read_pfm(paths["reference"]).shape
		if shape != (entry["height"], entry["width"], 3):
			parser.error(f"{name}: reference shape {shape} does not match manifest resolution")
		metadata = json.loads(paths["reference_metrics"].read_text())
		if metadata.get("width") != entry["width"] or metadata.get("height") != entry["height"]:
			parser.error(f"{name}: reference metrics resolution does not match manifest")
		references[name] = {
			"path": str(paths["reference"]),
			"metrics_path": str(paths["reference_metrics"]),
			"metrics_sha256": file_digest(paths["reference_metrics"]),
			"metrics": metadata,
		}
		commands[name] = [
			sys.executable,
			str(REPO / "scripts/benchmark.py"),
			str(paths["scene"]),
			"--binary",
			str(binary),
			"--reference",
			str(paths["reference"]),
			"--backends",
			*args.backends,
			"--samples",
			str(args.samples),
			"--repeats",
			str(args.repeats),
			"--warmups",
			str(args.warmups),
			"--width",
			str(entry["width"]),
			"--height",
			str(entry["height"]),
			"--exposure",
			str(entry["exposure"]),
			"--min-bounces",
			str(defaults["min_bounces"]),
			"--max-bounces",
			str(defaults["max_bounces"]),
			"--seed",
			str(defaults["seed"]),
			"--output-dir",
			str(destination / name),
		]
	if args.dry_run:
		print(json.dumps(commands, indent=2))
		return
	destination.mkdir(parents=True, exist_ok=True)
	report = {
		"schema_version": 1,
		"status": "running",
		"started_utc": datetime.now(timezone.utc).isoformat(),
		"manifest": str(manifest_path),
		"manifest_sha256": file_digest(manifest_path),
		"binary": str(binary),
		"binary_sha256": file_digest(binary),
		"settings": {
			"samples": args.samples,
			"repeats": args.repeats,
			"warmups": args.warmups,
			"backends": args.backends,
		},
		"requested_scenes": selected,
		"scenes": {},
	}

	def save():
		(destination / "suite-results.json").write_text(json.dumps(report, indent=2) + "\n")

	save()
	try:
		for name in selected:
			print(f"Qualifying {name}", flush=True)
			subprocess.run(commands[name], cwd=REPO, check=True)
			result_path = destination / name / "results.json"
			result = json.loads(result_path.read_text())
			if (
				result["binary_sha256"] != report["binary_sha256"]
				or not result["provenance_recheck"]["stable"]
			):
				raise RuntimeError(
					f"{name}: binary or asset provenance changed during qualification"
				)
			report["scenes"][name] = {
				"benchmark_results": str(result_path),
				"results_sha256": file_digest(result_path),
				"command": commands[name],
				"reference": references[name],
				"summary": result["summary"],
				"comparisons": result["comparisons"],
				"optix_speedup": result.get("optix_speedup", {}),
			}
			save()
		report["status"] = "complete"
	except Exception as error:
		report["status"] = "failed"
		report["error"] = str(error)
		raise
	finally:
		report["finished_utc"] = datetime.now(timezone.utc).isoformat()
		save()
	print(f"Suite results: {destination / 'suite-results.json'}")


if __name__ == "__main__":
	main()
