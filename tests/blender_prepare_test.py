#!/usr/bin/env python3
"""Focused preparation integration check; tiny CPU atlas bakes, no beauty renders."""

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import Exporter, arguments
from blender_prepare_scene import capture_render, prepare, split_material

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.world = None
bpy.ops.object.camera_add(location=(0, -6, 3))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (
	(Vector((0, 0, 0)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
)
scene.camera.data.dof.use_dof = True
scene.camera.data.dof.aperture_blades = 0
scene.camera.data.dof.focus_distance = 6
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.object
cube.name = "Mixed material cube"
supported = bpy.data.materials.new("Supported intact")
supported.use_nodes = True
supported.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.4, 0.7, 1)
cube.data.materials.append(supported)
original_uv = cube.data.uv_layers.active
for corner in original_uv.data:
	corner.uv = (0.25, 0.25)
unsupported = bpy.data.materials.new("Baked layered alpha emission")
unsupported.use_nodes = True
tree = unsupported.node_tree
shader = tree.nodes["Principled BSDF"]
out = tree.nodes["Material Output"]
shader.inputs["Base Color"].default_value = (0.3, 0.2, 0.1, 1)
shader.inputs["Emission Color"].default_value = (1, 0.1, 0.05, 1)
shader.inputs["Emission Strength"].default_value = 4
shader.inputs["Emission Strength"].default_value = 1
shader.inputs["Emission Strength"].keyframe_insert("default_value", frame=1)
shader.inputs["Emission Strength"].default_value = 4
shader.inputs["Emission Strength"].keyframe_insert("default_value", frame=7)
shader.inputs["Alpha"].default_value = 0.8
mix = tree.nodes.new("ShaderNodeMixShader")
mix.inputs[0].default_value = 0.75
transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
tree.links.new(transparent.outputs[0], mix.inputs[1])
tree.links.new(shader.outputs[0], mix.inputs[2])
tree.links.new(mix.outputs[0], out.inputs["Surface"])
added = tree.nodes.new("ShaderNodeAddShader")
black = tree.nodes.new("ShaderNodeBsdfTranslucent")
black.inputs["Color"].default_value = (0, 0, 0, 1)
tree.links.new(shader.outputs[0], added.inputs[0])
tree.links.new(black.outputs[0], added.inputs[1])
tree.links.new(added.outputs[0], mix.inputs[2])
noise = tree.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 2
tree.links.new(noise.outputs["Fac"], shader.inputs["Roughness"])
bump = tree.nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.1
tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
tree.links.new(bump.outputs[0], shader.inputs["Normal"])
# A nested group's implicit texture UV must remain the original render UV even
# after the unsupported material gets a completely different atlas UV.
image = bpy.data.images.new("Four-color UV test", width=2, height=2, alpha=False, float_buffer=True)
image.colorspace_settings.name = "Non-Color"
image.pixels[:] = [0.3, 0.2, 0.1, 1, 0.9, 0.1, 0.1, 1, 0.1, 0.8, 0.4, 1, 0.2, 0.3, 0.9, 1]
image.pack()
group = bpy.data.node_groups.new("Original UV texture", "ShaderNodeTree")
group.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
texture = group.nodes.new("ShaderNodeTexImage")
texture.image = image
group_output = group.nodes.new("NodeGroupOutput")
group.links.new(texture.outputs["Color"], group_output.inputs["Color"])
call = tree.nodes.new("ShaderNodeGroup")
call.node_tree = group
tree.links.new(call.outputs["Color"], shader.inputs["Base Color"])
normal = tree.nodes.new("ShaderNodeNormal")
normal.outputs["Normal"].default_value = (0.6, 0, 0.8)
tree.links.new(normal.outputs["Normal"], shader.inputs["Normal"])
cube.data.materials.append(unsupported)
for face in cube.data.polygons:
	face.material_index = face.index % 2
# A supported part from a nonzero source slot must also bind its sole material.
part = split_material(cube.data, 1)
assert len(part.polygons) == 3 and all(face.material_index == 0 for face in part.polygons)
bpy.data.meshes.remove(part)
# Render-only modifier must be evaluated, not rejected as viewport mismatch.
modifier = cube.modifiers.new("Render triangulation", "TRIANGULATE")
modifier.show_viewport = False
modifier.show_render = True
bpy.ops.object.light_add(type="SPOT", location=(0, -3, 4))
spot = bpy.context.object
spot.data.energy = 100
spot.rotation_euler = (Vector((0, 0, 0)) - spot.location).to_track_quat("-Z", "Y").to_euler()
scene.frame_set(7)
with tempfile.TemporaryDirectory(prefix="xenon_prepare_test_") as temporary:
	root = Path(temporary)
	args = arguments(
		["--output", str(root), "--width", "16", "--height", "16", "--samples", "1", "--instances"]
	)
	report = prepare(args, SimpleNamespace(bake_size=16, bake_world=False, max_objects=0))
	assert report["status"] == "ok" and report["depsgraph_mode"] == "RENDER"
	assert report["source_frame"] == 7
	assert report["baked_material_meshes"] == 1, report
	assert report["preserved_materials"] == ["Supported intact"], report
	assert supported.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value[0] > 0.19
	assert bpy.context.scene.camera.data.dof.use_dof
	assert bpy.context.scene.camera.data.dof.focus_distance == 6
	meshes = [o.data for o in bpy.context.scene.objects if o.type == "MESH"]
	assert sum(len(m.polygons) for m in meshes) == 12
	assert any(any(m == supported for m in mesh.materials) for mesh in meshes)
	scene_text = (root / "scene.xenon").read_text()
	assert "spotlight " in scene_text and "lens " in scene_text
	change = next(
		c["message"]
		for c in report["changes"]
		if isinstance(c["message"], dict) and "bakes" in c["message"]
	)
	bakes = change["bakes"]["channels"]
	print("BAKE RANGES", json.dumps(bakes), flush=True)
	assert set(bakes) == {"color", "roughness", "alpha", "normal", "emission"}, bakes
	assert (
		abs(bakes["color"]["maximum"] - 0.3) < 1e-5 and abs(bakes["color"]["minimum"] - 0.1) < 1e-5
	), bakes["color"]
	assert 0.59 < bakes["alpha"]["maximum"] < 0.61, bakes["alpha"]
	assert abs(bakes["emission"]["premultiplied_maximum"] - 2.4) < 1e-5, bakes["emission"]
	assert abs(bakes["emission"]["maximum"] - 4) < 1e-5, bakes["emission"]
	assert abs(bakes["alpha"]["maximum"] * bakes["emission"]["maximum"] - 2.4) < 1e-5
	assert bakes["normal"]["minimum"] < 0.2 and 0.85 < bakes["normal"]["maximum"] < 0.95, bakes[
		"normal"
	]
	prepared = next(
		material for mesh in meshes for material in mesh.materials if material != supported
	)
	node = next(n for n in prepared.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
	assert all(
		node.inputs[name].is_linked
		for name in ("Base Color", "Roughness", "Alpha", "Normal", "Emission Color")
	)
	assert unsupported.node_tree.nodes.get(mix.name), "Preparation changed source graph"
	assert not texture.inputs["Vector"].is_linked, "Preparation changed source node group"
	print(
		"PASS render snapshot, preserved material, nested UV, authored bake frame, alpha/emission/normal/scalar, camera DOF, spot approximation"
	)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.object.camera_add()
scene.camera = bpy.context.object
bpy.ops.mesh.primitive_cube_add()
prototype = bpy.context.object
prototype.hide_render = True
bpy.ops.mesh.primitive_plane_add()
emitter = bpy.context.object
group = bpy.data.node_groups.new("Render-density scatter", "GeometryNodeTree")
group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
modifier = emitter.modifiers.new("Scatter", "NODES")
modifier.node_group = group
output = group.nodes.new("NodeGroupOutput")
points = group.nodes.new("GeometryNodeMeshLine")
viewport = group.nodes.new("GeometryNodeIsViewport")
switch = group.nodes.new("GeometryNodeSwitch")
switch.input_type = "INT"
switch.inputs["True"].default_value = 9
switch.inputs["False"].default_value = 3
group.links.new(viewport.outputs[0], switch.inputs["Switch"])
group.links.new(switch.outputs[0], points.inputs["Count"])
info = group.nodes.new("GeometryNodeObjectInfo")
info.inputs["Object"].default_value = prototype
info.inputs["As Instance"].default_value = True
instances = group.nodes.new("GeometryNodeInstanceOnPoints")
group.links.new(points.outputs["Mesh"], instances.inputs["Points"])
group.links.new(info.outputs["Geometry"], instances.inputs["Instance"])
group.links.new(instances.outputs["Instances"], output.inputs["Geometry"])
with tempfile.TemporaryDirectory(prefix="xenon_prepare_scatter_") as temporary:
	args = arguments(["--output", temporary, "--width", "4", "--height", "4", "--samples", "1"])
	captured = capture_render(scene, Exporter(args))
	placements = [
		instance
		for instance in captured["instances"]
		if len(captured["meshes"][instance["key"]].polygons)
	]
	assert len(placements) == 3 and all(instance["instance"] for instance in placements), placements
	assert scene.render.engine != "XENON_PREPARE_CAPTURE"
	print(
		"PASS actual RENDER scatter has 3 placements versus 9 in viewport; hidden GN prototype retained"
	)

# A displacement-only incompatibility must not replace repeating material UVs
# with one small atlas over an entire tree/mesh. Group expansion is handled by
# Blender, with its original parameter bindings intact.
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.world = None
bpy.ops.object.camera_add(location=(0, -3, 2))
scene.camera = bpy.context.object
bpy.ops.mesh.primitive_plane_add()
plane = bpy.context.object
source_uv = [tuple(corner.uv) for corner in plane.data.uv_layers.active.data]
image = bpy.data.images.new(
	"Repeating material image", width=32, height=16, alpha=True, float_buffer=True
)
image.colorspace_settings.name = "Non-Color"
image.pixels[:] = [0.2, 0.3, 0.4, 0.6] * (32 * 16)
image.pack()
material = bpy.data.materials.new("Displacement only")
material.use_nodes = True
material.node_tree.nodes.clear()
output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
group = bpy.data.node_groups.new("Preserved PBR group", "ShaderNodeTree")
group.interface.new_socket(name="Surface", in_out="OUTPUT", socket_type="NodeSocketShader")
shader = group.nodes.new("ShaderNodeBsdfPrincipled")
texture = group.nodes.new("ShaderNodeTexImage")
texture.image = image
out = group.nodes.new("NodeGroupOutput")
group.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
group.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
group.links.new(shader.outputs[0], out.inputs["Surface"])
call = material.node_tree.nodes.new("ShaderNodeGroup")
call.node_tree = group
material.node_tree.links.new(call.outputs["Surface"], output.inputs["Surface"])
value = material.node_tree.nodes.new("ShaderNodeValue")
value.outputs[0].default_value = 0.2
material.node_tree.links.new(value.outputs[0], output.inputs["Displacement"])
plane.data.materials.append(material)
with tempfile.TemporaryDirectory(prefix="xenon_prepare_uv_preserve_") as temporary:
	root = Path(temporary)
	args = arguments(["--output", str(root), "--width", "8", "--height", "8", "--samples", "1"])
	report = prepare(args, SimpleNamespace(bake_size=16, bake_world=False, max_objects=0))
	assert report["baked_material_meshes"] == 0, report
	assert len(report["uv_preserved_simplified_materials"]) == 1, report
	prepared = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
	assert [tuple(corner.uv) for corner in prepared.data.uv_layers.active.data] == source_uv
	assert "XenonBakeUV" not in prepared.data.uv_layers
	assert output.inputs["Displacement"].is_linked and call.node_tree == group
	final = prepared.data.materials[0]
	assert any(node.type == "TEX_IMAGE" and node.image == image for node in final.node_tree.nodes)
	world = bpy.data.worlds.new("Unchanged world image")
	world.use_nodes = True
	world_tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
	world_tex.image = image
	bpy.context.scene.world = world
	from blender_prepare_scene import resize_surface_images

	resized = resize_surface_images(bpy.context.scene, root, 8)
	assert len(resized) == 1 and resized[0]["comparison_size"] == [8, 4]
	assert list(image.size) == [32, 16] and world_tex.image == image
	final_image = next(
		node.image
		for node in prepared.data.materials[0].node_tree.nodes
		if node.type == "TEX_IMAGE"
	)
	assert list(final_image.size) == [8, 4]
	assert abs(final_image.pixels[3] - 0.6) < 1e-4, "Resizing lost alpha"
	assert Path(resized[0]["path"]).read_bytes()[24] == 16, "Linear map precision was reduced"
	bpy.ops.wm.save_as_mainfile(filepath=str(root / "resized.blend"))
	bpy.ops.wm.open_mainfile(filepath=str(root / "resized.blend"))
	reloaded = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
	reloaded_image = next(
		node.image
		for node in reloaded.data.materials[0].node_tree.nodes
		if node.type == "TEX_IMAGE"
	)
	assert list(reloaded_image.size) == [8, 4], (
		"Saved comparison source reverted to original image dimensions"
	)
	print(
		"PASS displacement/group conversion preserves UV/alpha image without baking; optional cap retains alpha, PNG16 and saved dimensions; world unchanged"
	)

# Leaf assets commonly apply one opacity image inside Principled and again at
# an outer transparent mix, with a separate translucent lobe added in between.
# Cycles sums transparent extinction; the resulting coverage is A squared.
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.world = None
bpy.ops.object.camera_add(location=(0, 0, 3))
scene.camera = bpy.context.object
bpy.ops.mesh.primitive_plane_add(size=10)
plane = bpy.context.object
material = bpy.data.materials.new("Leaf double opacity")
material.use_nodes = True
tree = material.node_tree
shader = tree.nodes["Principled BSDF"]
out = tree.nodes["Material Output"]
image = bpy.data.images.new("Leaf opacity", width=8, height=8, alpha=False, float_buffer=True)
image.colorspace_settings.name = "Non-Color"
image.pixels[:] = [0.5, 0.5, 0.5, 1] * 64
image.pack()
texture = tree.nodes.new("ShaderNodeTexImage")
texture.image = image
coords = tree.nodes.new("ShaderNodeTexCoord")
mapping = tree.nodes.new("ShaderNodeMapping")
tree.links.new(coords.outputs["UV"], mapping.inputs["Vector"])
tree.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
add = tree.nodes.new("ShaderNodeAddShader")
mix = tree.nodes.new("ShaderNodeMixShader")
tree.links.new(texture.outputs["Color"], shader.inputs["Alpha"])
tree.links.new(texture.outputs["Color"], mix.inputs[0])
tree.links.new(shader.outputs[0], add.inputs[0])
tree.links.new(translucent.outputs[0], add.inputs[1])
tree.links.new(transparent.outputs[0], mix.inputs[1])
tree.links.new(add.outputs[0], mix.inputs[2])
tree.links.new(mix.outputs[0], out.inputs["Surface"])
plane.data.materials.append(material)
with tempfile.TemporaryDirectory(prefix="xenon_prepare_leaf_") as temporary:
	root = Path(temporary)
	scene.render.engine = "CYCLES"
	scene.cycles.device = "CPU"
	scene.cycles.samples = 256
	scene.cycles.use_denoising = False
	scene.render.resolution_x = scene.render.resolution_y = 8
	scene.render.resolution_percentage = 100
	scene.render.film_transparent = True
	scene.render.image_settings.file_format = "OPEN_EXR"
	scene.render.image_settings.color_mode = "RGBA"
	scene.render.image_settings.color_depth = "32"
	scene.render.filepath = str(root / "original-alpha.exr")
	bpy.ops.render.render(write_still=True)
	rendered = bpy.data.images.load(str(root / "original-alpha.exr"), check_existing=False)
	measured_alpha = sum(rendered.pixels[3::4]) / 64
	assert abs(measured_alpha - 0.25) < 0.025, measured_alpha
	args = arguments(["--output", str(root), "--width", "8", "--height", "8", "--samples", "1"])
	report = prepare(args, SimpleNamespace(bake_size=16, bake_world=False, max_objects=0))
	assert report["baked_material_meshes"] == 0, report
	prepared = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
	final = prepared.data.materials[0]
	shader = next(node for node in final.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
	alpha = shader.inputs["Alpha"].links[0].from_node.image
	assert abs(alpha.pixels[0] - 0.25) < 2e-5, alpha.pixels[0]
	assert "XenonBakeUV" not in prepared.data.uv_layers
	assert any("squared opacity" in str(c["message"]) for c in report["changes"])
	assert any("Identity Mapping" in str(c["message"]) for c in report["changes"])
	print(
		f"PASS Cycles measured original leaf coverage {measured_alpha:.6f}; prepared opacity .25, repeated UV preserved without atlas"
	)
