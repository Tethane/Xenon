#!/usr/bin/env python3
"""Import glTF/GLB/OBJ into Blender and author reproducible camera and illumination.

Run inside Blender: --python scripts/blender_import_scene.py -- --input path.gltf
--output build/fixtures/sponza [--camera-eye X Y Z --camera-target X Y Z]
Camera arguments use Blender's Z-up coordinates in source model units.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import bpy
import bmesh
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export import Exporter, arguments


def import_gltf_with_metadata(path):
	"""Retain alpha semantics from the exact importer material, without name guesses.

	Blender 4.5 uses the same DITHERED render setting for OPAQUE and MASK.
	Its importer mapping also handles duplicate names and vertex-color variants.
	This deliberately uses the pinned Blender importer, and restores its method
	even when an import fails.
	"""
	from io_scene_gltf2.blender.imp.material import BlenderMaterial

	original = BlenderMaterial.create

	def create(gltf, material_index, vertex_color):
		result = original(gltf, material_index, vertex_color)
		if material_index is not None:
			source = gltf.data.materials[material_index]
			material = bpy.data.materials[source.blender_material[vertex_color]]
			mode = source.alpha_mode or "OPAQUE"
			cutoff = 0.5 if source.alpha_cutoff is None else source.alpha_cutoff
			if mode not in ("OPAQUE", "MASK", "BLEND") or not math.isfinite(cutoff) or cutoff < 0:
				raise RuntimeError(f"Invalid glTF alpha contract for material {material_index}")
			material["xenon_alpha_mode"] = mode
			material["xenon_alpha_cutoff"] = cutoff
			material["xenon_gltf_material_index"] = material_index
		return result

	BlenderMaterial.create = staticmethod(create)
	try:
		result = bpy.ops.import_scene.gltf(filepath=str(path))
		if "FINISHED" not in result:
			raise RuntimeError(f"glTF import did not finish: {result}")
	finally:
		BlenderMaterial.create = staticmethod(original)


def frame_object_bounds(lower, upper, aspect, lens=27.0, sensor_width=36.0):
	"""Fit the bounding sphere inside both view axes, including portrait frames."""
	center = (lower + upper) * 0.5
	radius = max((upper - lower).length * 0.5, 1e-4)
	horizontal_half_angle = math.atan(sensor_width / (2 * lens))
	vertical_half_angle = math.atan(math.tan(horizontal_half_angle) / aspect)
	distance = 1.12 * radius / math.sin(min(horizontal_half_angle, vertical_half_angle))
	direction = Vector((-1.3, -1.8, 1.0)).normalized()
	return center + direction * distance, center


def main():
	argv = sys.argv[sys.argv.index("--") + 1 :]
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--input", required=True, type=Path)
	parser.add_argument(
		"--camera-mode",
		choices=("object", "interior"),
		default="object",
		help="Automatic camera: frame all geometry (default) or use the legacy courtyard interior view",
	)
	parser.add_argument("--camera-name", help="Use an authored imported perspective camera by name")
	parser.add_argument("--camera-eye", type=float, nargs=3)
	parser.add_argument("--camera-target", type=float, nargs=3)
	parser.add_argument(
		"--omit-material",
		action="append",
		default=[],
		help="Explicitly omit faces using this material from both comparison sources",
	)
	parser.add_argument(
		"--max-texture-size",
		type=int,
		default=0,
		help="Resize source images in both renderers; original downloads retained",
	)
	parser.add_argument("--unit-scale", type=float, default=1.0)
	custom, remaining = parser.parse_known_args(argv)
	if not math.isfinite(custom.unit_scale) or custom.unit_scale <= 0:
		parser.error("--unit-scale must be finite and positive")
	if custom.max_texture_size < 0:
		parser.error("--max-texture-size must be nonnegative")
	if custom.camera_target and not custom.camera_eye:
		parser.error("--camera-target requires --camera-eye")
	if custom.camera_name and custom.camera_eye:
		parser.error("--camera-name and --camera-eye are mutually exclusive")
	if any(
		not math.isfinite(v)
		for values in (custom.camera_eye, custom.camera_target)
		if values
		for v in values
	):
		parser.error("Camera coordinates must be finite")
	args = arguments(remaining)
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)
	path = custom.input.resolve()
	if path.suffix.lower() in (".gltf", ".glb"):
		import_gltf_with_metadata(path)
	elif path.suffix.lower() == ".obj":
		bpy.ops.wm.obj_import(filepath=str(path))
	else:
		raise RuntimeError("Import expects glTF, GLB, or OBJ; .blend is opened directly by Blender")
	changes = []
	for obj in list(bpy.context.scene.objects):
		if obj.type != "MESH":
			continue
		slots = {
			i
			for i, slot in enumerate(obj.material_slots)
			if slot.material
			and any(
				slot.material.name == name
				or (
					slot.material.name.startswith(name + ".")
					and slot.material.name[len(name) + 1 :].isdigit()
				)
				for name in custom.omit_material
			)
		}
		if slots:
			mesh = bmesh.new()
			mesh.from_mesh(obj.data)
			faces = [face for face in mesh.faces if face.material_index in slots]
			changes.append(
				{"object": obj.name, "omitted_faces": len(faces), "materials": custom.omit_material}
			)
			bmesh.ops.delete(mesh, geom=faces, context="FACES_ONLY")
			mesh.to_mesh(obj.data)
			mesh.free()
			if not obj.data.polygons:
				obj.hide_render = True
	# Some glTF primitives carry an all-white COLOR_0 attribute. Simplifying
	# that multiplication is exact and avoids requiring vertex-color transport.
	for material in bpy.data.materials:
		if not material.use_nodes:
			continue
		for node in list(material.node_tree.nodes):
			if node.type != "VERTEX_COLOR":
				continue
			valid = True
			found = False
			for obj in bpy.context.scene.objects:
				if obj.type != "MESH" or material not in [
					slot.material for slot in obj.material_slots
				]:
					continue
				attr = (
					obj.data.color_attributes.get(node.layer_name)
					if node.layer_name
					else obj.data.color_attributes.active_color
				)
				if attr is None:
					valid = False
					break
				values = np.empty(len(attr.data) * 4, dtype=np.float32)
				attr.data.foreach_get("color", values)
				if not np.all(np.abs(values - 1) < 1e-7):
					valid = False
					break
				found = True
			if valid and found:
				color = material.node_tree.nodes.new("ShaderNodeRGB")
				color.outputs[0].default_value = (1, 1, 1, 1)
				alpha = material.node_tree.nodes.new("ShaderNodeValue")
				alpha.outputs[0].default_value = 1
				for output in node.outputs:
					for link in list(output.links):
						material.node_tree.links.new(
							(alpha if output.name == "Alpha" else color).outputs[0], link.to_socket
						)
				changes.append(
					{
						"material": material.name,
						"exact_simplification": "All vertex colors verified (1,1,1,1); replaced attribute reads with that constant",
					}
				)
	if custom.max_texture_size:
		for image in bpy.data.images:
			width, height = image.size
			longest = max(width, height)
			if longest > custom.max_texture_size:
				ratio = custom.max_texture_size / longest
				image.scale(max(1, round(width * ratio)), max(1, round(height * ratio)))
				image.pack()
				changes.append(
					{
						"image": image.name,
						"original_size": [width, height],
						"comparison_size": list(image.size),
					}
				)
	scene = bpy.context.scene
	scene.unit_settings.system = "METRIC"
	scene.unit_settings.scale_length = custom.unit_scale
	scene.render.resolution_x = args.width or 800
	scene.render.resolution_y = args.height or 600
	scene.render.resolution_percentage = 100
	corners = [
		obj.matrix_world @ Vector(corner)
		for obj in scene.objects
		if obj.type == "MESH" and not obj.hide_render and len(obj.data.polygons)
		for corner in obj.bound_box
	]
	if not corners:
		raise RuntimeError("Imported source contains no mesh bounds for camera framing")
	lower = Vector(tuple(min(v[a] for v in corners) for a in range(3)))
	upper = Vector(tuple(max(v[a] for v in corners) for a in range(3)))
	center = (lower + upper) * 0.5
	size = upper - lower
	imported_cameras = [
		obj for obj in scene.objects if obj.type == "CAMERA" and obj.data.type == "PERSP"
	]
	authored = (
		bpy.data.objects.get(custom.camera_name)
		if custom.camera_name
		else (imported_cameras[0] if imported_cameras and not custom.camera_eye else None)
	)
	if custom.camera_name and (
		authored is None or authored.type != "CAMERA" or authored.data.type != "PERSP"
	):
		raise RuntimeError(f"Authored perspective camera not found: {custom.camera_name}")
	if authored:
		camera = authored
		eye = camera.matrix_world.translation.copy()
		target = eye - camera.matrix_world.col[2].to_3d().normalized()
	else:
		if custom.camera_eye:
			eye = Vector(custom.camera_eye)
			target = Vector(custom.camera_target or center)
		elif custom.camera_mode == "object":
			aspect = (
				scene.render.resolution_x
				* scene.render.pixel_aspect_x
				/ (scene.render.resolution_y * scene.render.pixel_aspect_y)
			)
			eye, target = frame_object_bounds(lower, upper, aspect)
		else:
			eye = Vector(
				(center.x - 0.32 * size.x, center.y - 0.08 * size.y, lower.z + 0.10 * size.z)
			)
			target = Vector((center.x + 0.18 * size.x, center.y, lower.z + 0.16 * size.z))
		bpy.ops.object.camera_add(location=eye)
		camera = bpy.context.object
		camera.name = "Xenon benchmark camera"
		camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
		camera.data.lens = 27
		camera.data.sensor_fit = "HORIZONTAL"
		camera.data.sensor_width = 36
		camera.data.clip_start = min(0.01 * custom.unit_scale, max(size.length * 1e-4, 1e-6))
		camera.data.clip_end = max(100000, (eye - center).length + size.length * 2)
	scene.camera = camera
	for obj in list(scene.objects):
		if obj.type == "LIGHT":
			changes.append(
				{
					"light": obj.name,
					"change": "Replaced imported light with the explicit benchmark rig",
				}
			)
			bpy.data.objects.remove(obj, do_unlink=True)
	# The explicit benchmark rig replaces imported lighting in the shared
	# .blend source for both renderers; every removed light is recorded above.
	world = bpy.data.worlds.new("Xenon courtyard daylight")
	world.use_nodes = True
	world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.28, 0.42, 0.62, 1)
	world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
	scene.world = world
	light = bpy.data.lights.new("Xenon late-afternoon sun", "SUN")
	light.energy = 3.0
	light.color = (1.0, 0.78, 0.52)
	light.angle = 0
	sun = bpy.data.objects.new(light.name, light)
	scene.collection.objects.link(sun)
	sun.rotation_euler = Vector((-0.5, 0.45, -0.65)).to_track_quat("-Z", "Y").to_euler()
	root = Path(args.output).resolve()
	root.mkdir(parents=True, exist_ok=True)
	report = {
		"source": str(path),
		"importer": "Blender " + bpy.app.version_string,
		"units_meters": custom.unit_scale,
		"camera_source": camera.name,
		"automatic_camera_mode": custom.camera_mode
		if not authored and not custom.camera_eye
		else None,
		"bounds_min": list(lower),
		"bounds_max": list(upper),
		"camera_eye": list(eye),
		"camera_target": list(target),
		"changes": changes,
		"lighting": "Authored benchmark rig: uniform blue world strength 0.22 plus ideal warm sun energy 3.0; imported light objects replaced and recorded",
	}
	(root / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
	bpy.ops.wm.save_as_mainfile(filepath=str(root / "source.blend"))
	Exporter(args).run()


if __name__ == "__main__":
	main()
