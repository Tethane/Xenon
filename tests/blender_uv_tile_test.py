#!/usr/bin/env python3
"""Repeated leaf cards retain their UVs and coverage when material graphs need baking."""

from pathlib import Path
import sys
import tempfile
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_prepare_scene import bake_material, uv_tile_layer

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.threads_mode = "FIXED"
bpy.context.scene.render.threads = 2
mesh = bpy.data.meshes.new("Repeated leaves")
vertices, faces = [], []
for index in range(64):
	x = index * 0.2
	base = len(vertices)
	vertices.extend([(x, 0, 0), (x + 0.1, 0, 0), (x + 0.1, 1, 0), (x, 1, 0)])
	faces.append(tuple(base + offset for offset in range(4)))
mesh.from_pydata(vertices, [], faces)
uv = mesh.uv_layers.new(name="LeafUV")
for index, loop in enumerate(uv.data):
	loop.uv = [(0, 0), (1, 0), (1, 1), (0, 1)][index % 4]
original_uvs = [tuple(loop.uv) for loop in uv.data]
obj = bpy.data.objects.new("Leaves", mesh)
bpy.context.scene.collection.objects.link(obj)
material = bpy.data.materials.new("Leaf color ramp")
material.use_nodes = True
mesh.materials.append(material)
tree = material.node_tree
shader = tree.nodes.get("Principled BSDF")
tex = tree.nodes.new("ShaderNodeTexImage")
ramp = tree.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.1, 0.03, 0, 1)
ramp.color_ramp.elements[1].color = (1, 0.3, 0, 1)
tree.links.new(tex.outputs["Color"], ramp.inputs[0])
tree.links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
tree.links.new(tex.outputs["Color"], shader.inputs["Alpha"])
with tempfile.TemporaryDirectory(prefix="xenon_leaf_uv_") as directory:
	root = Path(directory)
	image = bpy.data.images.new("Leaf coverage", width=16, height=16)
	pixels = np.ones((16, 16, 4), dtype=np.float32)
	pixels[:, :8, :3] = 0
	image.pixels.foreach_set(pixels.ravel())
	image.filepath_raw = str(root / "source.png")
	image.file_format = "PNG"
	image.save()
	tex.image = bpy.data.images.load(image.filepath_raw, check_existing=False)
	tex.image.colorspace_settings.name = "Non-Color"
	assert uv_tile_layer(material, mesh) == "LeafUV"
	uv.data[0].uv.x = -0.5
	assert uv_tile_layer(material, mesh) is None
	uv.data[0].uv.x = 0
	coordinates = tree.nodes.new("ShaderNodeTexCoord")
	tree.links.new(coordinates.outputs["Generated"], tex.inputs["Vector"])
	assert uv_tile_layer(material, mesh) is None
	tree.links.new(coordinates.outputs["UV"], tex.inputs["Vector"])
	assert uv_tile_layer(material, mesh) == "LeafUV"
	bump = tree.nodes.new("ShaderNodeBump")
	tree.links.new(tex.outputs["Color"], bump.inputs["Height"])
	tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
	assert uv_tile_layer(material, mesh) is None
	tree.nodes.remove(bump)
	changes = []
	result = bake_material(obj, material, root, 64, lambda where, message: changes.append(message))
	assert result != material
	assert len(mesh.polygons) == 64 and len(mesh.vertices) == 256
	assert [tuple(loop.uv) for loop in mesh.uv_layers["LeafUV"].data] == original_uvs
	assert len(mesh.uv_layers) == 1 and mesh.uv_layers.active.name == "LeafUV"
	assert any("uv_tile_bake" in change for change in changes)
	alpha_path = next(
		change["bakes"]["channels"]["alpha"]["path"] for change in changes if "bakes" in change
	)
	alpha = bpy.data.images.load(alpha_path, check_existing=False)
	alpha.colorspace_settings.name = "Non-Color"
	values = np.asarray(alpha.pixels[:]).reshape(64, 64, 4)[:, :, 0]
	assert np.all(values[:, 4:24] < 0.01), values[:, 4:24].max()
	assert np.all(values[:, 40:60] > 0.99), values[:, 40:60].min()
	assert 0.45 < values.mean() < 0.55, values.mean()
	assert shader.inputs["Base Color"].links[0].from_node == ramp
print("PASS: repeated leaf UVs and coverage preserved; geometry-dependent graphs rejected")
