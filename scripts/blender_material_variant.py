#!/usr/bin/env python3
"""Re-export materials from the unchanged Blender source while sharing mesh assets.

Run Blender with the same .blend used by the baseline:
  --python scripts/blender_material_variant.py -- --base-export build/fixtures/base
  --output build/fixtures/variant --compensated-materials

The baseline source checksum must match. Material identifiers and scene declarations
are verified. The variant shares the baseline meshes through a directory symlink;
keep both export directories together. No geometry is evaluated or regenerated.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export import Exporter, arguments, safe


def main():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--base-export", required=True, type=Path)
	custom, remaining = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
	args = arguments(remaining)
	base = custom.base_export.resolve()
	root = Path(args.output).resolve()
	if base == root:
		raise RuntimeError("Material variant must use a separate output directory")
	baseline = json.loads((base / "export-report.json").read_text())
	if baseline.get("status") != "ok":
		raise RuntimeError("Baseline export must have status ok")
	expected = baseline.get("source_sha256") or baseline.get("shared_source", {}).get("sha256")
	if not expected:
		raise RuntimeError(
			"Baseline lacks a shared-source SHA-256; re-export with the current exporter first"
		)
	with Path(bpy.data.filepath).open("rb") as source:
		source_sha = hashlib.file_digest(source, "sha256").hexdigest()
	if source_sha != expected:
		raise RuntimeError("Open Blender source differs from the baseline source checksum")
	root.mkdir(parents=True, exist_ok=True)
	scene_target = root / "scene.xenon"
	if scene_target.exists():
		scene_target.unlink()
	mesh_target = root / "meshes"
	if mesh_target.exists() or mesh_target.is_symlink():
		if not mesh_target.is_symlink() or mesh_target.resolve() != (base / "meshes").resolve():
			raise RuntimeError("Output mesh directory is not the expected baseline link")
	else:
		mesh_target.symlink_to(base / "meshes", target_is_directory=True)
	shutil.copytree(base / "textures", root / "textures", dirs_exist_ok=True)
	exporter = Exporter(args)
	scene = (base / "scene.xenon").read_text()
	for file in sorted(
		(base / "materials").glob("m*.mat"), key=lambda path: int(path.stem.split("_", 1)[0][1:])
	):
		# Only material files declared by the successful baseline are relevant;
		# failed older exports can leave unrelated files in the same directory.
		if f'matfile "materials/{file.name}"' not in scene.splitlines():
			continue
		name = file.stem.split("_", 1)[1]
		matches = [mat for mat in bpy.data.materials if safe(mat.name) == name]
		if len(matches) != 1:
			raise RuntimeError(f"Cannot uniquely bind source material for {file.name}")
		actual = exporter.material(matches[0])
		if actual != file.stem:
			raise RuntimeError(f"Material binding changed: {actual} vs {file.stem}")
	if exporter.report["errors"]:
		raise RuntimeError(exporter.report["errors"])
	declarations = [line for line in scene.splitlines() if line.startswith("matfile ")]
	actual = [line for line in exporter.lines if line.startswith("matfile ")]
	if actual != declarations:
		raise RuntimeError("Material declaration order or coverage differs from baseline")
	# Publish the scene only after every material validates successfully.
	scene_target.write_text(scene)
	report = dict(baseline)
	report["variant_mode"] = "materials_only"
	report["baseline_export_seconds"] = report.pop("export_seconds", None)
	report["approximations"] = exporter.report["approximations"]
	report["required_features"] = sorted(
		set(baseline.get("required_features", []) + exporter.report.get("required_features", []))
	)
	report["textures"] = list(
		{
			item["path"]: item
			for item in baseline.get("textures", []) + exporter.report["textures"]
		}.values()
	)
	report["notes"] = [
		*baseline.get("notes", []),
		*exporter.report["notes"],
		"Material-only variant shares the unchanged baseline scene and mesh assets; no geometry evaluation or modification.",
	]
	report["source_sha256"] = source_sha
	report["shared_source"] = {
		**baseline.get("shared_source", {}),
		"path": str(Path(bpy.data.filepath).resolve()),
		"sha256": source_sha,
		"baseline_scene": str(base / "scene.xenon"),
		"mesh_directory": str(base / "meshes"),
	}
	if (base / "cycles.pfm").exists():
		report["shared_source"]["cycles_reference"] = str(base / "cycles.pfm")
	(root / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
	print(
		f"EXPORTED material-only variant {root}: {len(exporter.materials)} materials, baseline geometry shared"
	)


if __name__ == "__main__":
	main()
