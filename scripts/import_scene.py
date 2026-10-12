#!/usr/bin/env python3
"""Import a local Blender or model file into a new Xenon scene directory.

Strict export is the default: unsupported features fail with export-report.json.
--prepare explicitly converts/bakes a .blend using the documented approximation.
glTF/GLB/OBJ import authors a benchmark camera/lighting rig and replaces imported
lights; it does not preserve their original illumination. See
docs/BLENDER_IMPORT.md for supported features, conversions and the rig.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def positive(value):
	try:
		number = int(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError("expected a positive integer") from error
	if not 0 < number <= 2_147_483_647:
		raise argparse.ArgumentTypeError("expected a positive 32-bit integer")
	return number


def object_name(value):
	if not value.strip() or "\0" in value:
		raise argparse.ArgumentTypeError("expected a nonempty object name")
	return value


def find_blender(explicit=None):
	selected = explicit if explicit is not None else os.environ.get("BLENDER")
	if selected is not None:
		found = shutil.which(str(Path(selected).expanduser())) if selected else None
		if not found:
			raise ValueError(
				f"Selected Blender executable is unavailable: {selected!r}; set --blender or BLENDER to an executable path."
			)
		return Path(found).resolve()
	for family in ("5.2", "4.5"):
		candidates = []
		for path in (ROOT / "build/tools").glob(f"blender-{family}.*/blender"):
			version = re.match(r"blender-(\d+)\.(\d+)\.(\d+)", path.parent.name)
			if version and path.is_file() and os.access(path, os.X_OK):
				candidates.append((tuple(map(int, version.groups())), path))
		if candidates:
			return max(candidates)[1].resolve()
	found = shutil.which("blender")
	if found:
		return Path(found).resolve()
	raise ValueError(
		"Blender was not found. Set --blender /path/to/blender, set BLENDER, install a local build/tools/blender-5.2.* or blender-4.5.*, or put blender on PATH."
	)


def digest(path):
	result = hashlib.sha256()
	with path.open("rb") as stream:
		while block := stream.read(1024 * 1024):
			result.update(block)
	return result.hexdigest()


def arguments(argv=None):
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		"source", type=Path, help="Existing .blend, .gltf, .glb or .obj file; not modified"
	)
	parser.add_argument(
		"--output",
		required=True,
		type=Path,
		help="New or empty export directory; existing contents are never overwritten",
	)
	parser.add_argument(
		"--blender",
		metavar="EXECUTABLE",
		help="Blender executable; then BLENDER, local 5.2/4.5, or PATH is used",
	)
	parser.add_argument(
		"--width",
		type=positive,
		help="Final image width; otherwise retain Blender resolution (model import defaults to 800)",
	)
	parser.add_argument(
		"--height",
		type=positive,
		help="Final image height; otherwise retain Blender resolution (model import defaults to 600)",
	)
	parser.add_argument(
		"--samples",
		type=positive,
		default=64,
		help="Xenon scene sample count (default: 64); this command does not render",
	)
	parser.add_argument(
		"--threads", type=positive, default=4, help="Blender worker threads (default: 4)"
	)
	parser.add_argument(
		"--instances",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Export deduplicated meshes and native placements (default: enabled)",
	)
	parser.add_argument(
		"--compensated-materials",
		action="store_true",
		help="Opt into the documented approximation for eligible opaque Blender MULTI_GGX materials",
	)
	parser.add_argument(
		"--prepare",
		action="store_true",
		help="Explicitly convert/bake a .blend into a separate comparison source; changes are recorded",
	)
	parser.add_argument(
		"--bake-size", type=positive, help="Texture bake dimension with --prepare (default: 256)"
	)
	parser.add_argument(
		"--bake-world",
		action="store_true",
		help="With --prepare, explicitly approximate the world as a 2048x1024 HDR environment using its diffuse-lighting branch",
	)
	parser.add_argument(
		"--volume-proxies",
		action="store_true",
		help="With --prepare, explicitly approximate unsupported fog by reported homogeneous boxes; changes spatial density and boundaries",
	)
	parser.add_argument(
		"--max-texture-size",
		type=positive,
		metavar="PIXELS",
		help="Explicitly resize large images in the derived scene; model imports or .blend with --prepare only; omitted preserves dimensions",
	)
	parser.add_argument(
		"--omit-area-lights",
		action="store_true",
		help="With --prepare, explicitly omit source area lamps instead of converting them to visible mesh emitters",
	)
	parser.add_argument(
		"--omit-object",
		action="append",
		type=object_name,
		default=[],
		metavar="NAME",
		help="With --prepare, explicitly omit this exact source object name; repeat for multiple objects",
	)
	args = parser.parse_args(argv)
	try:
		args.source = args.source.expanduser().resolve(strict=True)
		if not args.source.is_file():
			raise ValueError("Source must be a regular file")
		if args.source.suffix.lower() not in {".blend", ".gltf", ".glb", ".obj"}:
			raise ValueError("Source must end in .blend, .gltf, .glb or .obj")
		if args.prepare and args.source.suffix.lower() != ".blend":
			raise ValueError("--prepare is only supported for .blend inputs")
		if args.bake_size is not None and not args.prepare:
			raise ValueError("--bake-size requires --prepare")
		if args.bake_world and not args.prepare:
			raise ValueError("--bake-world requires --prepare")
		if args.volume_proxies and not args.prepare:
			raise ValueError("--volume-proxies requires --prepare")
		if (
			args.max_texture_size is not None
			and args.source.suffix.lower() == ".blend"
			and not args.prepare
		):
			raise ValueError(
				"--max-texture-size requires --prepare for .blend inputs; strict export preserves source images"
			)
		if (args.omit_area_lights or args.omit_object) and not args.prepare:
			raise ValueError(
				"--omit-area-lights and --omit-object require --prepare for a .blend input"
			)
		if ((args.width or 1) * (args.height or 1)) > 100_000_000:
			raise ValueError("Requested image exceeds Xenon's 100 million pixel limit")
		args.output = args.output.expanduser().resolve()
		if args.source.is_relative_to(args.output):
			raise ValueError("Output must not be the source file or a directory containing it")
		if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
			raise ValueError(
				"Output must be a new or empty directory. Use a different --output; inspect any previous failed export before removing it."
			)
		args.blender = find_blender(args.blender)
	except (OSError, ValueError) as error:
		parser.error(str(error))
	return args


def build_command(args):
	command = [str(args.blender), "--background", "--factory-startup", "--disable-autoexec"]
	if args.source.suffix.lower() == ".blend":
		command.append(str(args.source))
		script = "blender_prepare_scene.py" if args.prepare else "blender_export.py"
		source_args = []
	else:
		script = "blender_import_scene.py"
		source_args = ["--input", str(args.source)]
	command += [
		"--threads",
		str(args.threads),
		"--python-exit-code",
		"1",
		"--python",
		str(ROOT / "scripts" / script),
		"--",
		*source_args,
		"--output",
		str(args.output),
		"--samples",
		str(args.samples),
	]
	for key in ("width", "height"):
		if getattr(args, key) is not None:
			command += ["--" + key, str(getattr(args, key))]
	if args.instances:
		command.append("--instances")
	if args.compensated_materials:
		command.append("--compensated-materials")
	if args.max_texture_size is not None:
		command += ["--max-texture-size", str(args.max_texture_size)]
	if args.prepare:
		command += ["--bake-size", str(args.bake_size or 256)]
		if args.bake_world:
			command.append("--bake-world")
		if args.volume_proxies:
			command.append("--volume-proxies")
		if args.omit_area_lights:
			command.append("--omit-area-lights")
		command += ["--omit-object=" + name for name in args.omit_object]
	return command


def main(argv=None):
	args = arguments(argv)
	command = build_command(args)
	source_hash = digest(args.source)
	args.output.mkdir(parents=True, exist_ok=True)
	request_path = args.output / "import-request.json"
	request = {
		"schema_version": 1,
		"status": "running",
		"source": str(args.source),
		"source_sha256": source_hash,
		"output": str(args.output),
		"command": command,
		"prepare": args.prepare,
		"bake_world": args.bake_world,
		"instances": args.instances,
		"volume_proxies": args.volume_proxies,
		"max_texture_size": args.max_texture_size,
		"omit_area_lights": args.omit_area_lights,
		"omit_objects": args.omit_object,
		"compensated_materials": args.compensated_materials,
		"started_utc": datetime.now(timezone.utc).isoformat(),
	}
	request_path.write_text(json.dumps(request, indent=2) + "\n")
	if args.source.suffix.lower() != ".blend":
		print(
			"Model import authors a camera/lighting rig and replaces imported lights; see docs/BLENDER_IMPORT.md.",
			flush=True,
		)
	elif args.prepare:
		print(
			"Explicit preparation enabled: material conversion/baking is recorded in conversion-report.json; a separate comparison.blend is written.",
			flush=True,
		)
	print("Running: " + shlex.join(command), flush=True)
	try:
		result = subprocess.run(command, check=False)
		request["blender_exit_code"] = result.returncode
		request["source_sha256_after"] = digest(args.source)
		if request["source_sha256_after"] != source_hash:
			raise RuntimeError(
				"Source bytes changed during import; inspect the source and import-request.json"
			)
		if result.returncode:
			raise RuntimeError(
				f"Blender export failed (exit {result.returncode}). Inspect {args.output / 'export-report.json'} and Blender's output. No fallback conversion was attempted."
			)
		report = json.loads((args.output / "export-report.json").read_text())
		scene = args.output / "scene.xenon"
		if report.get("status") != "ok" or not scene.is_file() or not scene.stat().st_size:
			raise RuntimeError("Blender did not produce a successful export report and scene.xenon")
		request["status"] = "complete"
	except (OSError, ValueError, RuntimeError) as error:
		request["status"] = "failed"
		request["error"] = str(error)
		print(f"[Error] {error}", file=sys.stderr)
		return 1
	except KeyboardInterrupt:
		request["status"] = "interrupted"
		print(
			"[Error] Import interrupted; partial output was retained for inspection.",
			file=sys.stderr,
		)
		return 130
	finally:
		request["finished_utc"] = datetime.now(timezone.utc).isoformat()
		request_path.write_text(json.dumps(request, indent=2) + "\n")
	print(f"Exported: {scene}\nReport: {args.output / 'export-report.json'}")
	print(
		"Render from the repository root:\n  "
		+ shlex.join(["./xenon.sh", "--scene", str(scene), "--backend", "cpu"])
	)
	return 0


if __name__ == "__main__":
	try:
		sys.exit(main())
	except OSError as error:
		print(f"[Error] {error}", file=sys.stderr)
		sys.exit(1)
