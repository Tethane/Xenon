#!/usr/bin/env python3
"""Optional integration regression; run with the pinned Blender --python-exit-code 1.

blender --background --factory-startup --python-exit-code 1 --python tests/blender_export_test.py
"""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import Exporter


def args(path):
	return SimpleNamespace(output=str(path), width=64, height=64, samples=1, reference=False)


def scene():
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)
	bpy.ops.object.camera_add(location=(0, -5, 3))
	camera = bpy.context.object
	camera.rotation_euler = (
		(Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
	)
	bpy.context.scene.camera = camera
	bpy.context.scene.world = None
	bpy.context.scene.unit_settings.system = "METRIC"
	bpy.context.scene.unit_settings.scale_length = 1
	bpy.ops.mesh.primitive_plane_add(size=2)
	obj = bpy.context.object
	material = bpy.data.materials.new("Test")
	material.use_nodes = True
	obj.data.materials.append(material)
	return obj, material


with tempfile.TemporaryDirectory(prefix="xenon_blender_test_") as tmp:
	root = Path(tmp)
	obj, mat = scene()
	obj.scale = (-2, 1.5, 0.7)
	Exporter(args(root / "mirror")).run()
	vertices = []
	normals = []
	faces = []
	for line in (root / "mirror/meshes/scene.obj").read_text().splitlines():
		if line.startswith("v "):
			vertices.append(Vector(tuple(map(float, line.split()[1:]))))
		elif line.startswith("vn "):
			normals.append(Vector(tuple(map(float, line.split()[1:]))))
		elif line.startswith("f "):
			faces.append([int(word.split("/")[0]) - 1 for word in line.split()[1:]])
	assert len(faces) == 2
	for face in faces:
		a, b, c = (vertices[i] for i in face)
		assert (b - a).cross(c - a).dot(normals[face[0]]) > 0, (
			"Mirrored mesh winding disagrees with transformed normals"
		)
	duplicate = obj.copy()
	duplicate.data = obj.data
	bpy.context.scene.collection.objects.link(duplicate)
	duplicate.location = (3, 1, 2)
	duplicate.scale = (1, 2, 1)
	instance_args = args(root / "instances")
	instance_args.instances = True
	Exporter(instance_args).run()
	report = json.loads((root / "instances/export-report.json").read_text())
	assert (
		report["mesh_assets"] == 1
		and report["mesh_placements"] == 2
		and report["unique_triangles"] == 2
	)
	lines = (root / "instances/scene.xenon").read_text().splitlines()
	placements = [line.split()[2:] for line in lines if line.startswith("instance ")]
	local = [
		Vector(tuple(map(float, line.split()[1:])))
		for line in (root / "instances/meshes/asset_0000.obj").read_text().splitlines()
		if line.startswith("v ")
	]
	placed = []
	for placement in placements:
		values = list(map(float, placement))
		matrix = Matrix([values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]])
		placed.extend(tuple(round(v, 5) for v in matrix @ vertex) for vertex in local)
	Exporter(args(root / "flattened")).run()
	flat = [
		tuple(round(float(v), 5) for v in line.split()[1:])
		for line in (root / "flattened/meshes/scene.obj").read_text().splitlines()
		if line.startswith("v ")
	]
	assert sorted(placed) == sorted(flat), (
		"Native affine placements differ from flattened world geometry"
	)
	obj, mat = scene()
	image = bpy.data.images.new("Normal strength", width=2, height=2, float_buffer=True)
	image.colorspace_settings.name = "Non-Color"
	image.pixels = [0.7, 0.4, 0.5, 1] * 4
	image.pack()
	tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
	tex.image = image
	normal = mat.node_tree.nodes.new("ShaderNodeNormalMap")
	normal.inputs["Strength"].default_value = 0.2
	mat.node_tree.links.new(tex.outputs["Color"], normal.inputs["Color"])
	mat.node_tree.links.new(
		normal.outputs["Normal"], mat.node_tree.nodes["Principled BSDF"].inputs["Normal"]
	)
	Exporter(args(root / "normal")).run()
	derived = next((root / "normal/textures").glob("*_normal_strength_*.png"))
	result = bpy.data.images.load(str(derived))
	result.colorspace_settings.name = "Non-Color"
	red, green, blue, _ = result.pixels[:4]
	assert abs(red - 0.7) < 2e-5 and abs(green - 0.4) < 2e-5 and abs(blue - 0.9) < 2e-5, (
		red,
		green,
		blue,
	)
	compensated = args(root / "compensated")
	compensated.compensated_materials = True
	Exporter(compensated).run()
	assert (
		"multipleScattering 1" in next((root / "compensated/materials").glob("*.mat")).read_text()
	)
	mat.node_tree.nodes["Principled BSDF"].inputs["Transmission Weight"].default_value = 1
	compensated.output = str(root / "transmissive")
	Exporter(compensated).run()
	assert (
		"multipleScattering 1"
		not in next((root / "transmissive/materials").glob("*.mat")).read_text()
	)
	principled = mat.node_tree.nodes["Principled BSDF"]
	principled.inputs["Emission Color"].default_value = (1, 0.5, 0.2, 1)
	principled.inputs["Emission Strength"].default_value = 2
	Exporter(args(root / "principled-emission")).run()
	assert (
		"emissionTwoSided 1"
		in next((root / "principled-emission/materials").glob("*.mat")).read_text()
	)
	nodes = mat.node_tree.nodes
	nodes.clear()
	output = nodes.new("ShaderNodeOutputMaterial")
	emission = nodes.new("ShaderNodeEmission")
	mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
	Exporter(args(root / "mesh-emission")).run()
	assert (
		"emissionTwoSided 1" in next((root / "mesh-emission/materials").glob("*.mat")).read_text()
	)
	assert (
		"emissionTwoSided"
		in json.loads((root / "mesh-emission/export-report.json").read_text())["required_features"]
	)
	obj, mat = scene()
	obj.data.uv_layers.active.name = "Requested"
	other = obj.data.uv_layers.new(name="Wrong")
	other.active_render = True
	image = bpy.data.images.new("UV test", width=2, height=2)
	image.pixels = [1, 1, 1, 1] * 4
	image.pack()
	tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
	tex.image = image
	uv = mat.node_tree.nodes.new("ShaderNodeUVMap")
	uv.uv_map = "Requested"
	mat.node_tree.links.new(uv.outputs[0], tex.inputs["Vector"])
	mat.node_tree.links.new(
		tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
	)
	try:
		Exporter(args(root / "uv")).run()
	except RuntimeError:
		pass
	else:
		raise AssertionError("Named UV mismatch was silently accepted")
	report = json.loads((root / "uv/export-report.json").read_text())
	assert any("requires UV layer" in error["message"] for error in report["errors"])
	assert not (root / "uv/scene.xenon").exists()
	obj, mat = scene()
	bpy.ops.wm.save_as_mainfile(filepath=str(root / "source.blend"))
	Exporter(args(root / "variant-base")).run()
	from blender_material_variant import main as material_variant

	old_argv = sys.argv
	try:
		sys.argv = [
			"blender",
			"--",
			"--base-export",
			str(root / "variant-base"),
			"--output",
			str(root / "variant"),
			"--compensated-materials",
		]
		material_variant()
	finally:
		sys.argv = old_argv
	assert (root / "variant/meshes").resolve() == (root / "variant-base/meshes").resolve()
	assert (root / "variant/scene.xenon").read_bytes() == (
		root / "variant-base/scene.xenon"
	).read_bytes()
	assert "multipleScattering 1" in next((root / "variant/materials").glob("*.mat")).read_text()
	obj, mat = scene()
	sun = bpy.data.lights.new("IdentitySun", "SUN")
	sun.angle = 0
	sun.energy = 7
	sun.color = (0.2, 0.3, 0.4)
	sun.use_nodes = True
	sun.node_tree.nodes.get("Emission").inputs["Color"].default_value = (1, 1, 1, 1)
	sun.node_tree.nodes.get("Emission").inputs["Strength"].default_value = 1
	sun_obj = bpy.data.objects.new("IdentitySun", sun)
	bpy.context.scene.collection.objects.link(sun_obj)
	point = bpy.data.lights.new("IdentityPoint", "POINT")
	point.shadow_soft_size = 0
	point.energy = 4 * 3.141592653589793
	point.use_nodes = True
	point.node_tree.nodes.get("Emission").inputs["Color"].default_value = (1, 1, 1, 1)
	point.node_tree.nodes.get("Emission").inputs["Strength"].default_value = 1
	point_obj = bpy.data.objects.new("IdentityPoint", point)
	bpy.context.scene.collection.objects.link(point_obj)
	Exporter(args(root / "identity-lights")).run()
	lines = (root / "identity-lights/scene.xenon").read_text().splitlines()
	sunlight = next(line.split() for line in lines if line.startswith("sunlight "))
	pointlight = next(line.split() for line in lines if line.startswith("pointlight "))
	assert max(abs(float(a) - b) for a, b in zip(sunlight[4:], (0.2, 0.3, 0.4, 7))) < 1e-6
	assert max(abs(float(value) - 1) for value in pointlight[4:]) < 1e-6
	for case in ("strength", "color", "linked"):
		emission = sun.node_tree.nodes.get("Emission")
		emission.inputs["Strength"].default_value = 2 if case == "strength" else 1
		emission.inputs["Color"].default_value = (0.5, 1, 1, 1) if case == "color" else (1, 1, 1, 1)
		if case == "linked":
			value = sun.node_tree.nodes.new("ShaderNodeValue")
			value.outputs[0].default_value = 1
			sun.node_tree.links.new(value.outputs[0], emission.inputs["Strength"])
		try:
			Exporter(args(root / ("custom-light-" + case))).run()
		except RuntimeError:
			pass
		else:
			raise AssertionError("Custom light shader was silently accepted: " + case)
	obj, mat = scene()
	if hasattr(bpy.context.scene, "compositing_node_group"):
		bpy.context.scene.compositing_node_group = bpy.data.node_groups.new(
			"UnsupportedCompositor", "CompositorNodeTree"
		)
	else:
		bpy.context.scene.use_nodes = True
	try:
		Exporter(args(root / "compositor")).run()
	except RuntimeError:
		pass
	else:
		raise AssertionError("Assigned compositor was silently accepted")
	report = json.loads((root / "compositor/export-report.json").read_text())
	assert any("Compositing nodes" in error["message"] for error in report["errors"])
	assert not (root / "compositor/scene.xenon").exists()
print(
	"Blender exporter mirror/normal, native shared-instance, normal strength, and strict named-UV regressions passed"
)
