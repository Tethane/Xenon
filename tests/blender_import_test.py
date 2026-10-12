#!/usr/bin/env python3
"""CPU-only pinned-Blender regression for glTF alpha provenance and reference size."""

import base64
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_import_scene import import_gltf_with_metadata, frame_object_bounds
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from blender_export import render_reference


with tempfile.TemporaryDirectory(prefix="xenon_gltf_test_") as directory:
	root = Path(directory)
	# Repeated source names deliberately require the importer's exact mapping.
	payload = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
	gltf = {
		"asset": {"version": "2.0"},
		"buffers": [
			{
				"uri": "data:application/octet-stream;base64," + base64.b64encode(payload).decode(),
				"byteLength": len(payload),
			}
		],
		"bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(payload)}],
		"accessors": [
			{
				"bufferView": 0,
				"componentType": 5126,
				"count": 3,
				"type": "VEC3",
				"min": [0, 0, 0],
				"max": [1, 1, 0],
			}
		],
		"materials": [
			{
				"name": "Duplicate",
				"alphaMode": mode,
				"alphaCutoff": 0.3,
				"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 0.2]},
			}
			for mode in ("OPAQUE", "MASK", "BLEND")
		],
		"meshes": [
			{
				"primitives": [
					{"attributes": {"POSITION": 0}, "material": index} for index in range(3)
				]
			}
		],
		"nodes": [{"mesh": 0}],
		"scenes": [{"nodes": [0]}],
		"scene": 0,
	}
	path = root / "alpha.gltf"
	path.write_text(json.dumps(gltf))
	from io_scene_gltf2.blender.imp.material import BlenderMaterial

	original = BlenderMaterial.create
	import_gltf_with_metadata(path)
	assert BlenderMaterial.create is original, "Importer callback was not restored"
	found = {
		int(mat["xenon_gltf_material_index"]): mat
		for mat in bpy.data.materials
		if "xenon_gltf_material_index" in mat
	}
	assert len(found) == 3
	assert len({mat.name for mat in found.values()}) == 3
	for index, mode in enumerate(("OPAQUE", "MASK", "BLEND")):
		assert found[index]["xenon_alpha_mode"] == mode
		assert abs(found[index]["xenon_alpha_cutoff"] - 0.3) < 1e-7
	scene = bpy.context.scene
	scene.render.resolution_x = 12
	scene.render.resolution_y = 16
	scene.render.resolution_percentage = 50
	args = SimpleNamespace(
		width=None,
		height=None,
		reference_samples=1,
		samples=1,
		seed=1,
		bounces=2,
		cycles_device="CPU",
	)
	render_reference(scene, args, root)
	metadata = json.loads((root / "cycles-metrics.json").read_text())
	assert (metadata["width"], metadata["height"]) == (6, 8), metadata
	assert metadata["pixel_filter"] == "BOX"
	assert scene.cycles.pixel_filter_type == "BOX"

# Every bounding-box corner must remain in view for landscape and portrait.
for width, height in ((800, 600), (400, 900)):
	scene.render.resolution_x, scene.render.resolution_y = width, height
	scene.render.resolution_percentage = 100
	lower, upper = Vector((-2, -3, -1)), Vector((4, 1, 5))
	eye, target = frame_object_bounds(lower, upper, width / height)
	camera = scene.camera
	camera.location = eye
	camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
	camera.data.lens = 27
	camera.data.sensor_width = 36
	camera.data.sensor_fit = "HORIZONTAL"
	bpy.context.view_layer.update()
	for x in (lower.x, upper.x):
		for y in (lower.y, upper.y):
			for z in (lower.z, upper.z):
				ndc = world_to_camera_view(scene, camera, Vector((x, y, z)))
				assert 0 < ndc.x < 1 and 0 < ndc.y < 1 and ndc.z > 0, (width, height, ndc)

print("Blender glTF alpha, reference percentage and automatic framing regressions passed")
