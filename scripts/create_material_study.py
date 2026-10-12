#!/usr/bin/env python3
"""Derive an explicit Xenon material study from an exported studio scene.

The original comparison scene stays untouched. This intentionally changes the
BSDFs and glass absorption, so its renders are not matched Cycles comparisons.
Geometry/environment/image references resolve to the source assets; copied
material files and a dependency-hashed manifest record the reproducible variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex


def digest(path):
	result = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			result.update(block)
	return result.hexdigest()


def read_records(path):
	return [
		tokens
		for line in path.read_text().splitlines()
		if (tokens := shlex.split(line, comments=True))
	]


def quote(value):
	# Scene paths/names use C++ std::quoted syntax, with double-quote escaping.
	return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def derive(source, destination):
	source = source.resolve(strict=True)
	destination = destination.resolve()
	if destination == source.parent or source.parent.is_relative_to(destination):
		raise ValueError("Output must not contain or replace the original scene directory")
	records = read_records(source)
	if not records or records[0] != ["version", "2"]:
		raise ValueError("Material study requires an exported version 2 scene")
	dependencies = {}

	def dependency(path):
		path = path.resolve(strict=True)
		dependencies[str(path)] = digest(path)
		return path

	dependency(source)
	materials = []
	scene_lines = []
	roles = set()
	for tokens in records:
		command = tokens[0]
		if command == "matfile":
			original = dependency(source.parent / tokens[1])
			properties = read_records(original)
			values = {line[0]: line[1:] for line in properties}
			name = values.get("name", [original.stem])[0]
			changes = {}

			def scalar(key, default):
				return float(values.get(key, [str(default)])[0])

			def change(key, value, reason):
				changes[key] = {"before": values.get(key), "after": value.split(), "reason": reason}
				values[key] = value.split()

			opaque = scalar("transmission", 0) == 0
			isotropic = scalar("anisotropy", 0) == 0 and scalar("anisotropic", 0) == 0
			emissive = any(float(v) > 0 for v in values.get("emission", ["0", "0", "0"]))
			if opaque and isotropic and 1 <= scalar("ior", 1.5) <= 3 and not emissive:
				change(
					"multipleScattering",
					"1",
					"Enable validated opaque isotropic GGX energy compensation",
				)
			lower = name.lower()
			if "ceramic" in lower:
				if "multipleScattering" not in changes:
					raise ValueError(
						"Ceramic material must be an eligible opaque isotropic substrate"
					)
				roles.add("ceramic")
				change("diffuseRoughness", "0.35", "Add independent EON rough diffuse to ceramic")
				change("clearcoat", "0.18", "Add a subtle reciprocal approximate coating")
				change("clearcoatRoughness", "0.14", "Retain a soft coat highlight")
				change("clearcoatIor", "1.45", "Use moderate dielectric coat reflectance")
			if "floor" in lower:
				if "multipleScattering" not in changes:
					raise ValueError(
						"Floor material must be an eligible opaque isotropic substrate"
					)
				roles.add("floor")
				change(
					"diffuseRoughness",
					"0.65",
					"Use rough diffuse under the existing floor specular interface",
				)
			if "glass" in lower:
				if opaque or scalar("metallic", 0) >= 1:
					raise ValueError("Glass material must be transmissive dielectric")
				roles.add("glass")
				change(
					"absorption",
					"0.42 0.08 0.035",
					"Introduce distance-dependent cool glass tint, coefficients in inverse meters",
				)
			output_name = f"{len(materials):03d}_{original.name}"
			lines = []
			seen = set()
			for key, *old in properties:
				if key in seen:
					raise ValueError(f"Duplicate material key {key!r} in {original}")
				seen.add(key)
				value = values[key]
				if key.endswith("Texture"):
					image = dependency(original.parent / value[0])
					lines.append(key + " " + quote(image) + " " + " ".join(value[1:]))
				elif key in ("name", "type"):
					lines.append(key + " " + quote(value[0]))
				else:
					lines.append(" ".join([key, *value]))
			for key in changes:
				if key not in seen:
					lines.append(" ".join([key, *values[key]]))
			materials.append(
				{
					"name": name,
					"source": str(original),
					"output": "materials/" + output_name,
					"changes": changes,
					"text": "\n".join(lines) + "\n",
				}
			)
			scene_lines.append("matfile " + quote("materials/" + output_name))
		elif command in ("mesh", "environment"):
			path = dependency(source.parent / tokens[1])
			scene_lines.append(
				command
				+ " "
				+ quote(path)
				+ (" " + " ".join(tokens[2:]) if len(tokens) > 2 else "")
			)
		else:
			scene_lines.append(
				" ".join(
					quote(token) if any(c.isspace() for c in token) else token for token in tokens
				)
			)
	if roles != {"ceramic", "floor", "glass"}:
		raise ValueError(
			f"Source is missing required studio materials: {sorted({'ceramic', 'floor', 'glass'} - roles)}"
		)
	(destination / "materials").mkdir(parents=True, exist_ok=True)
	for material in materials:
		(destination / material["output"]).write_text(material.pop("text"))
	(destination / "scene.xenon").write_text("\n".join(scene_lines) + "\n")
	manifest = {
		"format": "xenon-native-material-study-v1",
		"source": str(source),
		"purpose": "Native material beauty study; deliberately different from the matched Cycles source",
		"scene": "scene.xenon",
		"materials": materials,
		"dependencies_sha256": dependencies,
		"limitations": [
			"Compensation and EON use the documented Xenon approximations, not exact Blender Principled",
			"Reciprocal coat omits internal layer returns and can lose energy in a white furnace",
			"Glass absorption requires closed oriented boundaries; nested relative IOR is not modeled",
			"Geometry, environment and images are absolute references: retain the listed source dependencies",
		],
	}
	(destination / "material-study.json").write_text(json.dumps(manifest, indent=2) + "\n")
	return manifest


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--source", type=Path, default=Path("build/fixtures/hdr_studio/scene.xenon")
	)
	parser.add_argument("--output", type=Path, default=Path("build/fixtures/energy_studio"))
	args = parser.parse_args()
	manifest = derive(args.source, args.output)
	print(
		json.dumps(
			{
				"scene": str(args.output.resolve() / "scene.xenon"),
				"modified_materials": sum(bool(m["changes"]) for m in manifest["materials"]),
				"manifest": str(args.output.resolve() / "material-study.json"),
			},
			indent=2,
		)
	)


if __name__ == "__main__":
	main()
