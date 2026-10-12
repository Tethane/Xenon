#!/usr/bin/env python3
"""Run inside Blender; measures procedural density and verifies disjoint proxies."""

import json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_prepare_volume import prepare_volumes, partition, _density_samples, _boundary
from blender_export import Exporter, arguments

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.frame_set(5)


def material(name, density, color, g, procedural=False):
	m = bpy.data.materials.new(name)
	m.use_nodes = True
	t = m.node_tree
	t.nodes.clear()
	v = t.nodes.new("ShaderNodeVolumePrincipled")
	v.inputs["Color"].default_value = (*color, 1)
	v.inputs["Density"].default_value = density
	v.inputs["Anisotropy"].default_value = g
	v.inputs["Absorption Color"].default_value = (0.25, 0.25, 0.25, 1)
	out = t.nodes.new("ShaderNodeOutputMaterial")
	t.links.new(v.outputs[0], out.inputs["Volume"])
	if procedural:
		tex = t.nodes.new("ShaderNodeTexCoord")
		sep = t.nodes.new("ShaderNodeSeparateXYZ")
		t.links.new(tex.outputs["Generated"], sep.inputs[0])
		t.links.new(sep.outputs["X"], v.inputs["Density"])
	return m


bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.object
cube.name = "Constant exact box"
cube.data.materials.append(material("Constant Principled", 0.2, (0.3, 0.5, 0.7), 0.2))
bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1, location=(1, 0, 0))
sphere = bpy.context.object
sphere.name = "Procedural sphere"
sphere.data.materials.append(material("Measured gradient", 1, (1, 1, 1), -0.3, True))
# Hidden collection volumes must not be resurrected as visible proxies.
hidden = bpy.data.collections.new("Render-hidden fog")
scene.collection.children.link(hidden)
hidden.hide_render = True
bpy.ops.mesh.primitive_cube_add(size=2, location=(10, 0, 0))
hidden_obj = bpy.context.object
hidden_obj.data.materials.append(material("Hidden Principled", 100, (1, 0, 0), 0))
for collection in list(hidden_obj.users_collection):
	collection.objects.unlink(hidden_obj)
hidden.objects.link(hidden_obj)
scene.view_layers.update()
report = prepare_volumes(scene, "build/volume-proxy-test", grid_size=8)
assert report["status"] == "converted" and len(report["regions"]) == 2
r = next(x for x in report["regions"] if x["object"] == "Constant exact box")
assert all(abs(a - b) < 1e-6 for a, b in zip(r["sigma_s"], (0.06, 0.10, 0.14)))
assert all(abs(a - b) < 1e-6 for a, b in zip(r["sigma_a"], (0.07, 0.05, 0.03)))
r = next(x for x in report["regions"] if x["object"] == "Procedural sphere")
assert abs(r["mean_inside_density"] - 0.5) < 0.02, r
assert 0.4 < r["occupancy"] < 0.6, r
assert abs(r["density_in_proxy"] - r["mean_inside_density"] * r["occupancy"]) < 1e-12
cells = report["partition"]
for i, a in enumerate(cells):
	for b in cells[i + 1 :]:
		assert not all(
			min(a["upper"][k], b["upper"][k]) > max(a["lower"][k], b["lower"][k]) + 1e-8
			for k in range(3)
		)
assert any(len(c["sources"]) == 2 for c in cells)
# Coefficient-volume integrals must be conserved by overlap partition.
for field in ("sigma_a", "sigma_s"):
	for k in range(3):
		before = sum(
			x[field][k] * math.prod(x["upper"][a] - x["lower"][a] for a in range(3))
			for x in report["regions"]
		)
		after = sum(
			x[field][k] * math.prod(x["upper"][a] - x["lower"][a] for a in range(3)) for x in cells
		)
		assert abs(before - after) < 1e-8
assert cube.hide_render and sphere.hide_render
# Unconnected procedural Vector defaults must use the original Generated
# coordinates, rather than the sampling mesh's UV-tile bounding box.
implicit = material("Implicit Generated", 1, (1, 1, 1), 0)
tree = implicit.node_tree
gradient = tree.nodes.new("ShaderNodeTexGradient")
volume = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeVolumePrincipled")
tree.links.new(gradient.outputs["Fac"], volume.inputs["Density"])
measured = _density_samples(
	sphere,
	implicit,
	volume.name,
	[(0.5, 0, 0), (1.5, 0, 0)],
	Path("build/volume-proxy-test/implicit.exr").resolve(),
)
assert max(abs(measured[0] - 0.25), abs(measured[1] - 0.75)) < 0.01, measured
# A locally reversed face must fail even though total signed volume is positive.
bad_mesh = bpy.data.meshes.new("Mixed winding")
faces = [list(face.vertices) for face in cube.data.polygons]
faces[0].reverse()
bad_mesh.from_pydata([tuple(vertex.co) for vertex in cube.data.vertices], [], faces)
bad_obj = bpy.data.objects.new("Mixed winding", bad_mesh)
scene.collection.objects.link(bad_obj)
try:
	_boundary(bad_obj)
except ValueError as error:
	assert "consistent face winding" in str(error)
else:
	raise AssertionError("Mixed volume winding was accepted")
bpy.data.objects.remove(bad_obj, do_unlink=True)
bpy.data.meshes.remove(bad_mesh)
camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.location = (4, -6, 4)
camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
scene.view_layers.update()
args = arguments(
	[
		"--output",
		"build/volume-proxy-test/export",
		"--width",
		"32",
		"--height",
		"32",
		"--samples",
		"1",
		"--instances",
	]
)
exporter = Exporter(args)
exporter.run()
assert not exporter.report["errors"], exporter.report["errors"]
print(
	"VOLUME_PROXY_OK",
	json.dumps(
		{
			"cells": len(cells),
			"measured_mean": r["mean_inside_density"],
			"occupancy": r["occupancy"],
			"grid": 8,
		}
	),
)
