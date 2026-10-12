#!/usr/bin/env python3
"""Check material-study provenance, resource rebasing and intentional BSDF edits."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_material_study import derive, digest, read_records


with tempfile.TemporaryDirectory(prefix="xenon material study ") as temp:
	root = Path(temp)
	source = root / "source"
	(source / "materials").mkdir(parents=True)
	(source / "meshes").mkdir()
	(source / "textures").mkdir()
	(source / "meshes/scene.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
	(source / "textures/map.png").write_bytes(b"path-only texture fixture")
	for name in ("ceramic", "floor", "glass", "emitter"):
		extra = (
			"transmission 1\n"
			if name == "glass"
			else "emission 2 3 4\n"
			if name == "emitter"
			else ""
		)
		if name == "ceramic":
			extra += 'baseColorTexture "../textures/map.png" srgb\n'
		(source / f"materials/{name}.mat").write_text(f'version 2\nname "{name}"\nior 1.5\n{extra}')
	scene = source / "scene.xenon"
	scene.write_text(
		"version 2\n"
		+ "".join(
			f'matfile "materials/{name}.mat"\n' for name in ("ceramic", "floor", "glass", "emitter")
		)
		+ 'mesh "meshes/scene.obj"\n'
	)
	original = {path: digest(path) for path in source.rglob("*") if path.is_file()}
	destination = root / "variant"
	manifest = derive(scene, destination)
	assert all(digest(path) == expected for path, expected in original.items()), "Source changed"
	materials = {
		entry["name"]: dict(
			(line[0], line[1:]) for line in read_records(destination / entry["output"])
		)
		for entry in manifest["materials"]
	}
	assert materials["ceramic"]["multipleScattering"] == ["1"]
	assert materials["ceramic"]["clearcoat"] == ["0.18"]
	assert materials["floor"]["diffuseRoughness"] == ["0.65"]
	assert materials["glass"]["absorption"] == ["0.42", "0.08", "0.035"]
	assert "multipleScattering" not in materials["glass"]
	assert "multipleScattering" not in materials["emitter"]
	assert Path(materials["ceramic"]["baseColorTexture"][0]) == source / "textures/map.png"
	assert dict((line[0], line[1:]) for line in read_records(destination / "scene.xenon"))[
		"mesh"
	] == [str(source / "meshes/scene.obj")]
	assert manifest == json.loads((destination / "material-study.json").read_text())
	before = {path: digest(path) for path in destination.rglob("*") if path.is_file()}
	derive(scene, destination)
	assert all(digest(path) == expected for path, expected in before.items()), (
		"Variant is not reproducible"
	)
	try:
		derive(scene, source)
	except ValueError:
		pass
	else:
		raise AssertionError("Variant overwrote its source directory")

print("Native material-study paths, changes, immutable sources and reproducibility passed")
