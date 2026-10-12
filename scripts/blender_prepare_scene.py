#!/usr/bin/env python3
"""Create a reported, explicit Blender comparison source from render-time geometry.

Supported materials are retained. Unsupported surfaces are converted individually
with scene-linear color/scalar/coverage/emission and tangent-normal bakes.
UV-only materials retain their coordinates; other graphs use mesh atlases.
This is a documented approximation, not universal Blender shader translation.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import struct
import sys
import time
import zlib
from types import SimpleNamespace
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export import Exporter, arguments, safe


def scalar(node, name, default):
	socket = node.inputs.get(name)
	if not socket:
		return default
	if not socket.is_linked:
		return float(socket.default_value)
	source = socket.links[0].from_node
	return float(source.outputs[0].default_value) if source.type == "VALUE" else default


def flatten_groups(material):
	"""Let Blender preserve group interfaces/defaults while ungrouping a copy."""
	tree = material.node_tree

	def groups():
		reached = set()

		def visit(node):
			if node in reached:
				return
			reached.add(node)
			for socket in node.inputs:
				if socket.is_linked:
					visit(socket.links[0].from_node)

		for node in tree.nodes:
			if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
				visit(node)
		return [node for node in tree.nodes if node in reached and node.type == "GROUP"]

	if not groups():
		return 0
	area = bpy.context.screen.areas[0]
	old_type = area.type
	old_tree = area.spaces.active.node_tree if old_type == "NODE_EDITOR" else None
	count = 0
	try:
		area.type = "NODE_EDITOR"
		space = area.spaces.active
		space.tree_type = "ShaderNodeTree"
		space.shader_type = "OBJECT"
		space.pin = True
		space.path.start(tree)
		with bpy.context.temp_override(
			area=area, region=next(r for r in area.regions if r.type == "WINDOW")
		):
			while groups():
				node = groups()[0]
				if count >= 4096:
					raise RuntimeError(f"{material.name}: shader group expansion limit reached")
				for item in tree.nodes:
					item.select = False
				node.select = True
				tree.nodes.active = node
				if bpy.ops.node.group_ungroup() != {"FINISHED"}:
					raise RuntimeError(f"{material.name}: group expansion failed")
				count += 1
	finally:
		area.type = old_type
		if old_tree:
			area.spaces.active.path.start(old_tree)
	return count


def simplify_material(original, output_dir=None):
	"""Try a local, reported conversion before sacrificing an existing UV layout."""
	material = original.copy()
	material.name = safe(original.name) + "_prepared_graph"
	changes = []
	if not material.use_nodes:
		material.use_nodes = True
		changes.append("Non-node material replaced with explicit default Principled surface")
	tree = material.node_tree
	for output in tree.nodes:
		if output.type == "OUTPUT_MATERIAL" and output.is_active_output:
			if output.inputs["Displacement"].is_linked:
				for link in list(output.inputs["Displacement"].links):
					tree.links.remove(link)
				changes.append(
					"Shader displacement omitted; evaluated geometry and existing Normal/Bump graph retained"
				)
	count = flatten_groups(material)
	if count:
		changes.append(
			f"{count} shader groups expanded by Blender; linked inputs and interface defaults preserved"
		)
	for node in list(tree.nodes):
		if (
			node.type != "MAPPING"
			or node.vector_type not in ("POINT", "TEXTURE", "VECTOR")
			or not node.inputs["Vector"].is_linked
		):
			continue
		neutral = {"Location": (0, 0, 0), "Rotation": (0, 0, 0), "Scale": (1, 1, 1)}
		if any(
			node.inputs[name].is_linked
			or any(abs(a - b) > 1e-8 for a, b in zip(node.inputs[name].default_value, value))
			for name, value in neutral.items()
		):
			continue
		incoming = node.inputs["Vector"].links[0].from_socket
		for link in list(node.outputs["Vector"].links):
			tree.links.new(incoming, link.to_socket)
		changes.append(f"Identity Mapping {node.name!r} bypassed without changing coordinates")
	outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
	shader = (
		outputs[0].inputs["Surface"].links[0].from_node
		if len(outputs) == 1 and outputs[0].inputs["Surface"].is_linked
		else None
	)
	while shader and shader.type == "REROUTE" and shader.inputs[0].is_linked:
		shader = shader.inputs[0].links[0].from_node
	if shader and shader.type in ("MIX_SHADER", "ADD_SHADER"):
		reached = set()

		def visit(node):
			if node in reached:
				return
			reached.add(node)
			for socket in node.inputs:
				if socket.is_linked:
					visit(socket.links[0].from_node)

		visit(shader)
		emission = False
		for node in reached:
			if node.type == "EMISSION":
				emission = True
			if node.type == "BSDF_PRINCIPLED":
				color = node.inputs.get("Emission Color") or node.inputs.get("Emission")
				strength = node.inputs.get("Emission Strength")
				if (
					color
					and (color.is_linked or any(v != 0 for v in color.default_value[:3]))
					and (strength is None or strength.is_linked or strength.default_value != 0)
				):
					emission = True
		if not emission:
			selected, _ = choose(material)
			opacity = surface_coverage(tree, outputs[0].inputs["Surface"])
			opacity = fold_squared_opacity(tree, opacity, output_dir, changes)
			if selected.type != "EMISSION" and opacity is not None:
				prior = shader.bl_idname
				if selected.type != "BSDF_PRINCIPLED":
					replacement = tree.nodes.new("ShaderNodeBsdfPrincipled")
					for source_name, target_name in (
						("Color", "Base Color"),
						("Roughness", "Roughness"),
						("IOR", "IOR"),
						("Normal", "Normal"),
					):
						source = selected.inputs.get(source_name)
						target = replacement.inputs[target_name]
						if source:
							if source.is_linked:
								tree.links.new(source.links[0].from_socket, target)
							else:
								target.default_value = source.default_value
					replacement.inputs["Metallic"].default_value = (
						1 if selected.type in ("BSDF_GLOSSY", "BSDF_ANISOTROPIC") else 0
					)
					replacement.inputs["Transmission Weight"].default_value = (
						1 if selected.type == "BSDF_GLASS" else 0
					)
					if selected.type == "BSDF_DIFFUSE":
						replacement.inputs["IOR"].default_value = 1
					selected = replacement
				for link in list(selected.inputs["Alpha"].links):
					tree.links.remove(link)
				if isinstance(opacity, (int, float)):
					selected.inputs["Alpha"].default_value = opacity
				else:
					tree.links.new(opacity, selected.inputs["Alpha"])
				tree.links.new(selected.outputs[0], outputs[0].inputs["Surface"])
				shader = selected
				material["xenon_alpha_mode"] = "BLEND"
				changes.append(
					f"{prior} replaced by selected Principled lobe with original scalar coverage; other diffuse/translucent/layer lobes and colored transparent tint omitted. Nonemissive graph only; original image maps retained when valid."
				)
	if shader and shader.type == "BSDF_PRINCIPLED":
		rough = shader.inputs.get("Roughness")
		if rough and rough.is_linked:
			linked = rough.links[0].from_socket
			if linked.node.type == "TEX_IMAGE" and linked.name == "Color":
				separate = tree.nodes.new("ShaderNodeSeparateColor")
				separate.mode = "RGB"
				tree.links.new(linked, separate.inputs[0])
				tree.links.new(separate.outputs["Red"], rough)
				changes.append(
					"Image-driven roughness uses the red channel explicitly; colored roughness maps may differ from Blender luminance conversion"
				)
		neutral = {
			"Subsurface Weight": 0,
			"Coat Weight": 0,
			"Sheen Weight": 0,
			"Anisotropic IOR Level": 0,
			"Anisotropic Rotation": 0,
			"Specular IOR Level": 0.5,
			"Coat IOR": 1.5,
			"Thin Film Thickness": 0,
		}
		for name, value in neutral.items():
			socket = shader.inputs.get(name)
			if socket and (socket.is_linked or abs(socket.default_value - value) > 1e-6):
				prior = (
					socket.links[0].from_node.bl_idname
					if socket.is_linked
					else float(socket.default_value)
				)
				for link in list(socket.links):
					tree.links.remove(link)
				socket.default_value = value
				changes.append(f"{name} changed from {prior} to {value}")
		tint = shader.inputs.get("Specular Tint")
		if tint and (tint.is_linked or any(abs(v - 1) > 1e-6 for v in tint.default_value[:3])):
			for link in list(tint.links):
				tree.links.remove(link)
			tint.default_value = (1, 1, 1, 1)
			changes.append("Specular tint made neutral white")
	if (
		shader
		and shader.type in ("BSDF_GLASS", "BSDF_GLOSSY", "BSDF_ANISOTROPIC")
		and shader.distribution not in ("GGX", "MULTI_GGX")
	):
		changes.append(f"{shader.distribution} distribution converted to GGX")
		shader.distribution = "GGX"
	if shader and shader.type == "BSDF_GLASS" and shader.distribution == "MULTI_GGX":
		shader.distribution = "GGX"
		changes.append("Multiple-scattering glass converted to GGX single-scattering glass")
	material["xenon_prepared_graph"] = True
	return material, changes


def surface_coverage(tree, socket):
	"""Coverage algebra with constant folding; unsupported expressions still bake."""
	if not socket.is_linked:
		return 0.0
	node = socket.links[0].from_node
	if node.type == "REROUTE":
		return surface_coverage(tree, node.inputs[0])
	if node.type == "BSDF_TRANSPARENT":
		return 0.0
	if node.type == "BSDF_PRINCIPLED":
		alpha = node.inputs["Alpha"]
		return (
			alpha.links[0].from_socket
			if alpha.is_linked
			else max(0, min(1, float(alpha.default_value)))
		)

	def same(a, b):
		return (
			a == b
			if isinstance(a, (int, float)) and isinstance(b, (int, float))
			else not isinstance(a, (int, float))
			and not isinstance(b, (int, float))
			and a.as_pointer() == b.as_pointer()
		)

	def math_node(operation, a, b):
		if isinstance(a, (int, float)) and isinstance(b, (int, float)):
			return {
				"ADD": lambda: a + b,
				"SUBTRACT": lambda: a - b,
				"MULTIPLY": lambda: a * b,
				"MINIMUM": lambda: min(a, b),
			}[operation]()
		if operation == "MULTIPLY":
			if isinstance(a, (int, float)) and a in (0, 1):
				return 0.0 if a == 0 else b
			if isinstance(b, (int, float)) and b in (0, 1):
				return 0.0 if b == 0 else a
		if operation == "ADD":
			if isinstance(a, (int, float)) and a == 0:
				return b
			if isinstance(b, (int, float)) and b == 0:
				return a
		value = tree.nodes.new("ShaderNodeMath")
		value.operation = operation
		for target, item in zip(value.inputs, (a, b)):
			if isinstance(item, (int, float)):
				target.default_value = item
			else:
				tree.links.new(item, target)
		return value.outputs[0]

	if node.type in ("MIX_SHADER", "ADD_SHADER"):
		offset = 1 if node.type == "MIX_SHADER" else 0
		a, b = (
			surface_coverage(tree, node.inputs[offset]),
			surface_coverage(tree, node.inputs[offset + 1]),
		)
		if a is None or b is None:
			return None
		if node.type == "ADD_SHADER":
			# Add Shader sums transparent extinction, not normalized lobes:
			# alpha = 1 - ((1-a)+(1-b)) = a+b-1. Keep signed intermediate
			# coverage; only the final Principled alpha is clamped.
			if isinstance(a, (int, float)) and a == 1:
				return b
			if isinstance(b, (int, float)) and b == 1:
				return a
			return math_node("SUBTRACT", math_node("ADD", a, b), 1.0)
		if same(a, b):
			return a
		factor = node.inputs[0]
		f = (
			factor.links[0].from_socket
			if factor.is_linked
			else max(0, min(1, float(factor.default_value)))
		)
		if isinstance(f, (int, float)) and f in (0, 1):
			return a if f == 0 else b
		if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == 0 and b == 1:
			return f
		return math_node(
			"ADD",
			math_node("MULTIPLY", math_node("SUBTRACT", 1.0, f), a),
			math_node("MULTIPLY", f, b),
		)
	if node.type in (
		"BSDF_DIFFUSE",
		"BSDF_GLASS",
		"BSDF_GLOSSY",
		"BSDF_ANISOTROPIC",
		"BSDF_TRANSLUCENT",
		"EMISSION",
	):
		return 1.0
	return None


def fold_squared_opacity(tree, opacity, output_dir, changes):
	"""Retain leaf UVs when a nested transparent graph uses the same alpha twice."""
	if opacity is None or isinstance(opacity, (int, float)):
		return opacity
	multiply = opacity.node
	if (
		multiply.type != "MATH"
		or multiply.operation != "MULTIPLY"
		or not all(s.is_linked for s in multiply.inputs[:2])
	):
		return opacity
	left, right = (s.links[0].from_socket for s in multiply.inputs[:2])
	if left.as_pointer() != right.as_pointer() or left.node.type != "TEX_IMAGE":
		return opacity
	source = left.node
	image = source.image
	if not image or left.name not in ("Color", "Alpha"):
		return opacity
	if left.name == "Color" and image.colorspace_settings.name not in (
		"Non-Color",
		"Linear",
		"Linear Rec.709",
	):
		return opacity
	width, height = image.size
	pixels = np.empty(len(image.pixels), dtype=np.float32)
	image.pixels.foreach_get(pixels)
	if not width or not height or not len(pixels):
		return opacity
	pixels = pixels.reshape((height, width, image.channels))
	if left.name == "Color":
		if image.channels < 3 or np.any(np.abs(pixels[:, :, :3] - pixels[:, :, :1]) > 1e-7):
			return opacity
		values = pixels[:, :, 0]
	else:
		if image.channels < 4:
			return opacity
		values = pixels[:, :, 3]
	if not np.all(np.isfinite(values)):
		raise RuntimeError(f"{image.name}: nonfinite opacity values")
	values = np.clip(values, 0, 1) ** 2
	derived = bpy.data.images.new(
		safe(image.name) + "_coverage_squared",
		width=width,
		height=height,
		alpha=False,
		float_buffer=True,
	)
	data = np.ones((height, width, 4), dtype=np.float32)
	data[:, :, :3] = values[:, :, None]
	derived.pixels.foreach_set(data.ravel())
	derived.colorspace_settings.name = "Non-Color"
	if output_dir is not None:
		directory = Path(output_dir) / "material-textures"
		directory.mkdir(parents=True, exist_ok=True)
		path = directory / (safe(image.name) + "_coverage_squared.png")
		encoded = np.rint(data[::-1, :, :3] * 65535).astype(">u2")

		def chunk(kind, blob):
			return (
				struct.pack(">I", len(blob))
				+ kind
				+ blob
				+ struct.pack(">I", zlib.crc32(kind + blob) & 0xFFFFFFFF)
			)

		path.write_bytes(
			b"\x89PNG\r\n\x1a\n"
			+ chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
			+ chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row.tobytes() for row in encoded)))
			+ chunk(b"IEND", b"")
		)
		bpy.data.images.remove(derived)
		derived = bpy.data.images.load(str(path), check_existing=True)
		derived.colorspace_settings.name = "Non-Color"
	texture = tree.nodes.new("ShaderNodeTexImage")
	texture.image = derived
	texture.extension = source.extension
	texture.interpolation = source.interpolation
	texture.projection = source.projection
	if source.inputs["Vector"].is_linked:
		tree.links.new(source.inputs["Vector"].links[0].from_socket, texture.inputs["Vector"])
	changes.append(
		f"Doubled transparent coverage converted to squared opacity image {derived.name!r} at {width}x{height}, retaining UVs; filtering the squared texels approximates squaring filtered opacity at texture edges"
	)
	return texture.outputs["Color"]


def resize_surface_images(scene, root, limit):
	"""Resize copies of surface images; keep authored world lighting resolution."""
	if not limit:
		return []
	destination = root / "resized-textures"
	destination.mkdir(exist_ok=True)
	images = {}
	materials = {}
	changes = []

	def visit(tree):
		for node in tree.nodes:
			if node.type == "GROUP" and node.node_tree:
				node.node_tree = node.node_tree.copy()
				visit(node.node_tree)
			if node.type != "TEX_IMAGE" or node.image is None:
				continue
			original = node.image
			key = original.as_pointer()
			if max(original.size) <= limit:
				continue
			if key not in images:
				width, height = original.size
				ratio = limit / max(width, height)
				resized = original.copy()
				resized.scale(max(1, round(width * ratio)), max(1, round(height * ratio)))
				# Save new bytes before packing: copying a packed image otherwise
				# retains the old packed resolution in the comparison .blend.
				pixels = None
				if original.is_float and original.colorspace_settings.name != "sRGB":
					pixels = np.empty(len(resized.pixels), dtype=np.float32)
					resized.pixels.foreach_get(pixels)
					pixels = pixels.reshape((resized.size[1], resized.size[0], resized.channels))
					if not np.all(np.isfinite(pixels)):
						raise RuntimeError(f"{original.name}: nonfinite resized image")
				suffix = (
					".hdr"
					if pixels is not None and (np.any(pixels < 0) or np.any(pixels > 1))
					else ".png"
				)
				path = destination / (f"{len(images):04d}_" + safe(original.name) + suffix)
				if pixels is not None and suffix == ".png":
					channels = 4 if resized.channels >= 4 else 3
					values = np.rint(pixels[::-1, :, :channels] * 65535).astype(">u2")

					def chunk(kind, data):
						return (
							struct.pack(">I", len(data))
							+ kind
							+ data
							+ struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
						)

					path.write_bytes(
						b"\x89PNG\r\n\x1a\n"
						+ chunk(
							b"IHDR",
							struct.pack(
								">IIBBBBB",
								resized.size[0],
								resized.size[1],
								16,
								6 if channels == 4 else 2,
								0,
								0,
								0,
							),
						)
						+ chunk(
							b"IDAT",
							zlib.compress(b"".join(b"\0" + row.tobytes() for row in values)),
						)
						+ chunk(b"IEND", b"")
					)
				else:
					if suffix == ".hdr" and (
						np.any(pixels < 0)
						or (resized.channels >= 4 and np.any(pixels[:, :, 3] != 1))
					):
						raise RuntimeError(
							f"{original.name}: resized HDR image has negative values or alpha that RGBE cannot preserve"
						)
					resized.filepath_raw = str(path)
					resized.file_format = "HDR" if suffix == ".hdr" else "PNG"
					resized.save()
				loaded = bpy.data.images.load(str(path), check_existing=False)
				loaded.colorspace_settings.name = original.colorspace_settings.name
				loaded.alpha_mode = original.alpha_mode
				images[key] = loaded
				changes.append(
					{
						"image": original.name,
						"original_size": [width, height],
						"comparison_size": list(loaded.size),
						"path": str(path),
						"color_space": loaded.colorspace_settings.name,
					}
				)
				bpy.data.images.remove(resized)
			node.image = images[key]

	seen = set()
	for obj in scene.objects:
		if obj.type != "MESH" or obj.data.as_pointer() in seen:
			continue
		seen.add(obj.data.as_pointer())
		for index, material in enumerate(obj.data.materials):
			if material is None or not material.use_nodes:
				continue
			key = material.as_pointer()
			if key not in materials:
				copied = material.copy()
				visit(copied.node_tree)
				materials[key] = copied
			obj.data.materials[index] = materials[key]
	return changes


def choose(material):
	nodes = material.node_tree.nodes
	output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
	reached = set()

	def visit(node):
		if node in reached:
			return
		reached.add(node)
		for socket in node.inputs:
			if socket.is_linked:
				visit(socket.links[0].from_node)

	if output.inputs["Surface"].is_linked:
		visit(output.inputs["Surface"].links[0].from_node)
	for kind in (
		"BSDF_PRINCIPLED",
		"BSDF_DIFFUSE",
		"BSDF_GLASS",
		"GROUP",
		"BSDF_GLOSSY",
		"BSDF_ANISOTROPIC",
		"EMISSION",
	):
		for node in nodes:
			if (
				node in reached
				and node.type == kind
				and (node.inputs.get("Color") or node.inputs.get("Base Color"))
			):
				return node, output
	raise RuntimeError(f"{material.name}: no supported closure or exposed Color group found")


def capture_render(source, validator, omit_objects=(), omit_area_lights=False):
	"""Copy evaluated geometry while Blender's RENDER dependency graph is alive."""
	result = {
		"instances": [],
		"lights": [],
		"meshes": {},
		"error": None,
		"camera": None,
		"omitted": {},
	}
	omit_objects = set(omit_objects)
	missing = omit_objects - {obj.name for obj in source.objects}
	if missing:
		raise RuntimeError(
			"Requested omitted objects were not found: " + ", ".join(sorted(missing))
		)
	if source.camera and source.camera.name in omit_objects:
		raise RuntimeError("Cannot omit the active camera")

	class Capture(bpy.types.RenderEngine):
		bl_idname = "XENON_PREPARE_CAPTURE"
		bl_label = "Xenon render geometry capture"
		bl_use_preview = False

		def render(self, depsgraph):
			try:
				if depsgraph.mode != "RENDER":
					raise RuntimeError("Expected render dependency graph")
				validator.validate_evaluation(depsgraph)
				if validator.report["errors"]:
					raise RuntimeError(json.dumps(validator.report["errors"]))
				camera = source.camera.evaluated_get(depsgraph) if source.camera else None
				if camera:
					data = camera.data.copy()
					if data.dof.focus_object:
						focus = data.dof.focus_object.evaluated_get(
							depsgraph
						).matrix_world.translation
						data.dof.focus_distance = abs(
							(focus - camera.matrix_world.translation).dot(
								-camera.matrix_world.col[2].to_3d().normalized()
							)
						)
						data.dof.focus_object = None
					result["camera"] = (data, camera.matrix_world.copy())
				for inst in depsgraph.object_instances:
					if not Exporter.instance_visible(inst):
						continue
					obj = inst.object
					name = obj.original.name
					if name in omit_objects or (
						omit_area_lights and obj.type == "LIGHT" and obj.data.type == "AREA"
					):
						result["omitted"][name] = result["omitted"].get(name, 0) + 1
						continue
					if obj.type in ("CAMERA", "EMPTY", "ARMATURE"):
						continue
					matrix = inst.matrix_world.copy()
					if obj.type == "LIGHT":
						result["lights"].append((obj.name, obj.data.copy(), matrix))
						continue
					if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
						raise RuntimeError(
							f"{obj.name}: unsupported render object {obj.type}; no geometry omitted"
						)
					# Evaluated material handles expire with the render graph;
					# bind their stable source datablocks to copied mesh data.
					materials = tuple(
						slot.material.original if slot.material else None
						for slot in obj.material_slots
					)
					key = (
						obj.data.as_pointer(),
						tuple(m.as_pointer() if m else 0 for m in materials),
					)
					if key not in result["meshes"]:
						mesh = bpy.data.meshes.new_from_object(
							obj, preserve_all_data_layers=True, depsgraph=depsgraph
						)
						if mesh is None:
							raise RuntimeError(f"{obj.name}: render mesh unavailable")
						indices = [face.material_index for face in mesh.polygons]
						mesh.materials.clear()
						for material in materials:
							mesh.materials.append(material)
						for face, index in zip(mesh.polygons, indices):
							face.material_index = index
						result["meshes"][key] = mesh
					flags = [
						name
						for name in (
							"visible_camera",
							"visible_diffuse",
							"visible_glossy",
							"visible_transmission",
							"visible_volume_scatter",
							"visible_shadow",
						)
						if not getattr(obj, name, True)
					]
					if obj.is_holdout:
						flags.append("is_holdout")
					if obj.is_shadow_catcher:
						flags.append("is_shadow_catcher")
					result["instances"].append(
						{
							"name": obj.name,
							"key": key,
							"matrix": matrix,
							"instance": inst.is_instance,
							"visibility_changes": flags,
						}
					)
			except Exception as error:
				result["error"] = error
			image = self.begin_result(0, 0, 1, 1)
			self.end_result(image)

	settings = {
		name: getattr(source.render, name)
		for name in (
			"engine",
			"resolution_x",
			"resolution_y",
			"resolution_percentage",
			"use_compositing",
			"use_sequencer",
			"use_border",
		)
	}
	bpy.utils.register_class(Capture)
	try:
		source.render.engine = Capture.bl_idname
		source.render.resolution_x = source.render.resolution_y = 4
		source.render.resolution_percentage = 100
		source.render.use_compositing = False
		source.render.use_sequencer = False
		source.render.use_border = False
		bpy.ops.render.render()
	finally:
		for name, value in settings.items():
			setattr(source.render, name, value)
		bpy.utils.unregister_class(Capture)
	if result["error"]:
		raise result["error"]
	if result["camera"] is None:
		raise RuntimeError("Active camera required")
	return result


def split_material(mesh, index):
	"""Preserve corner normals, UV seams, and shader attributes during face split."""
	faces = [face for face in mesh.polygons if face.material_index == index]
	result = bpy.data.meshes.new(mesh.name + "_slot_" + str(index))
	# Retain unused positions: Generated coordinates use the original mesh bounds.
	result.from_pydata(
		[vertex.co[:] for vertex in mesh.vertices], [], [tuple(face.vertices) for face in faces]
	)
	loops = [loop for face in faces for loop in face.loop_indices]
	for layer in mesh.uv_layers:
		target = result.uv_layers.new(name=layer.name)
		target.data.foreach_set("uv", [value for i in loops for value in layer.data[i].uv])
		target.active_render = layer.active_render
	if mesh.uv_layers.active:
		result.uv_layers.active = result.uv_layers[mesh.uv_layers.active.name]
	normals = [mesh.corner_normals[i].vector[:] for i in loops]
	for old, new in zip(faces, result.polygons):
		new.use_smooth = old.use_smooth
	if normals:
		result.normals_split_custom_set(normals)
	for attribute in mesh.attributes:
		if (
			attribute.name.startswith(".")
			or attribute.name == "material_index"
			or attribute.name in result.attributes
		):
			continue
		if attribute.domain not in ("POINT", "CORNER", "FACE"):
			continue
		mapping = (
			range(len(mesh.vertices))
			if attribute.domain == "POINT"
			else loops
			if attribute.domain == "CORNER"
			else [face.index for face in faces]
		)
		try:
			target = result.attributes.new(attribute.name, attribute.data_type, attribute.domain)
			field = {
				"FLOAT": "value",
				"INT": "value",
				"BOOLEAN": "value",
				"FLOAT_VECTOR": "vector",
				"FLOAT_COLOR": "color",
				"BYTE_COLOR": "color",
				"FLOAT2": "vector",
			}.get(attribute.data_type)
			if not field:
				raise RuntimeError(f"Unsupported attribute type {attribute.data_type}")
			for dst, src in zip(target.data, mapping):
				setattr(dst, field, getattr(attribute.data[src], field))
		except Exception as error:
			raise RuntimeError(
				f"{mesh.name}: cannot retain attribute {attribute.name}: {error}"
			) from error
	return result


def is_volume(material):
	if not material or not material.use_nodes:
		return False
	return any(
		node.type == "OUTPUT_MATERIAL" and node.is_active_output and node.inputs["Volume"].is_linked
		for node in material.node_tree.nodes
	)


def pin_uv(material, name):
	if not name:
		return
	copies = {}

	def visit(tree):
		uv = tree.nodes.new("ShaderNodeUVMap")
		uv.uv_map = name
		for node in list(tree.nodes):
			if node.type == "TEX_IMAGE" and not node.inputs["Vector"].is_linked:
				tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
			if node.type == "TEX_COORD":
				for link in list(node.outputs["UV"].links):
					tree.links.new(uv.outputs["UV"], link.to_socket)
			if node.type == "NORMAL_MAP" and node.space == "TANGENT" and not node.uv_map:
				node.uv_map = name
			if node.type == "GROUP" and node.node_tree:
				key = node.node_tree.as_pointer()
				if key not in copies:
					copies[key] = node.node_tree.copy()
					visit(copies[key])
				node.node_tree = copies[key]

	visit(material.node_tree)


def uv_tile_layer(material, mesh):
	"""Return a UV layer only when a unit-tile bake is independent of geometry.

	Repeated leaf cards must not each consume a tiny island in a whole-tree
	atlas. Keep their existing coordinates and evaluate the material once on a
	UV square. Restrict this path to known nodes and coordinates inside the tile;
	object/generated coordinates, bump derivatives and UDIMs need mesh baking.
	"""
	uv = next((layer for layer in mesh.uv_layers if layer.active_render), mesh.uv_layers.active)
	if not uv or not material.use_nodes:
		return None
	coordinates = np.empty(len(uv.data) * 2, dtype=np.float32)
	uv.data.foreach_get("uv", coordinates)
	if not coordinates.size or not np.all(np.isfinite(coordinates)):
		return None
	if coordinates.min() < 0 or coordinates.max() > 1:
		return None
	allowed = {
		"OUTPUT_MATERIAL",
		"BSDF_PRINCIPLED",
		"BSDF_DIFFUSE",
		"BSDF_GLOSSY",
		"BSDF_ANISOTROPIC",
		"BSDF_GLASS",
		"BSDF_TRANSPARENT",
		"EMISSION",
		"MIX_SHADER",
		"ADD_SHADER",
		"TEX_IMAGE",
		"TEX_COORD",
		"UVMAP",
		"NORMAL_MAP",
		"MAPPING",
		"MATH",
		"VECT_MATH",
		"VALUE",
		"RGB",
		"VALTORGB",
		"MIX_RGB",
		"MIX",
		"HUE_SAT",
		"BRIGHTCONTRAST",
		"GAMMA",
		"INVERT",
		"RGBTOBW",
		"SEPARATE_COLOR",
		"COMBINE_COLOR",
		"SEPXYZ",
		"COMBXYZ",
		"CLAMP",
		"MAP_RANGE",
		"REROUTE",
	}
	seen = set()

	def visit(node):
		if node in seen:
			return True
		seen.add(node)
		if node.type not in allowed:
			return False
		if node.type == "TEX_IMAGE" and (
			not node.image or node.image.source != "FILE" or node.projection != "FLAT"
		):
			return False
		if node.type == "TEX_COORD" and node.from_instancer:
			return False
		if node.type == "UVMAP" and (node.from_instancer or node.uv_map not in ("", uv.name)):
			return False
		if node.type == "NORMAL_MAP" and (
			node.space != "TANGENT" or node.uv_map not in ("", uv.name)
		):
			return False
		for socket in node.inputs:
			for link in socket.links:
				if link.from_node.type == "TEX_COORD" and link.from_socket.name != "UV":
					return False
				if not visit(link.from_node):
					return False
		return True

	outputs = [
		n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
	]
	if len(outputs) != 1 or outputs[0].inputs["Displacement"].is_linked:
		return None
	return uv.name if visit(outputs[0]) else None


def bake_material(obj, original, root, size, record):
	uv_name = uv_tile_layer(original, obj.data)
	if not uv_name:
		return bake_material_surface(obj, original, root, size, record)

	mesh = bpy.data.meshes.new(obj.name + "_uv_tile")
	mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
	uv = mesh.uv_layers.new(name=uv_name)
	for loop, coordinate in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
		loop.uv = coordinate
	tile = bpy.data.objects.new(obj.name + "_uv_tile", mesh)
	bpy.context.scene.collection.objects.link(tile)
	try:
		material = bake_material_surface(tile, original, root, size, record, uv_name=uv_name)
		obj.data.materials.clear()
		obj.data.materials.append(material)
		obj.data.uv_layers.active = obj.data.uv_layers[uv_name]
		obj.data.uv_layers.active.active_render = True
		record(
			obj.name,
			{
				"uv_tile_bake": {
					"material": original.name,
					"uv_layer": uv_name,
					"size": size,
					"geometry_and_uvs": "preserved",
					"method": "UV-only graph evaluated on one unit square; repeated surface UVs share the baked texture",
				}
			},
		)
		return material
	finally:
		bpy.data.objects.remove(tile, do_unlink=True)
		bpy.data.meshes.remove(mesh)


def bake_material_surface(obj, original, root, size, record, uv_name=None):
	mesh = obj.data
	material = original.copy()
	mesh.materials.clear()
	mesh.materials.append(material)
	material.name = safe(obj.name) + "_prepared"
	if not material.use_nodes:
		material.use_nodes = True
	shader, output = choose(material)
	shader_type = shader.type
	color = shader.inputs.get("Base Color") or shader.inputs.get("Color")
	roughness = scalar(shader, "Roughness", 0.5)
	metallic = (
		1.0 if shader.type in ("BSDF_GLOSSY", "BSDF_ANISOTROPIC") else scalar(shader, "Metallic", 0)
	)
	transmission = 1.0 if shader.type == "BSDF_GLASS" else scalar(shader, "Transmission Weight", 0)
	ior = scalar(shader, "IOR", 1.5)
	if shader.type == "BSDF_DIFFUSE":
		ior = 1.0
	if shader.type == "EMISSION":
		ior = 1.0
	old_uv = next((uv for uv in mesh.uv_layers if uv.active_render), mesh.uv_layers.active)
	pin_uv(material, old_uv.name if old_uv else None)
	for other in bpy.context.scene.objects:
		other.select_set(False)
	obj.hide_render = False
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj
	if uv_name is None:
		mesh.uv_layers.new(name="XenonBakeUV")
		mesh.uv_layers.active = mesh.uv_layers["XenonBakeUV"]
		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.mesh.select_all(action="SELECT")
		bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.025)
		bpy.ops.object.mode_set(mode="OBJECT")
		mesh.uv_layers.active = mesh.uv_layers["XenonBakeUV"]
		mesh.uv_layers.active.active_render = True
		uv_name = "XenonBakeUV"
	work = root / "bake-work"
	work.mkdir(exist_ok=True)
	baked = root / "baked"
	baked.mkdir(exist_ok=True)
	base = safe(obj.name)
	channels = [{"kind": "color", "socket": color.name}, {"kind": "alpha"}, {"kind": "emission"}]
	for name in ("Roughness", "Metallic"):
		socket = shader.inputs.get(name)
		if socket and socket.is_linked:
			channels.append({"kind": name.lower(), "socket": name})
	if any(
		n.inputs.get("Normal") and n.inputs["Normal"].is_linked for n in material.node_tree.nodes
	):
		channels.append({"kind": "normal"})
	for channel in channels:
		channel["path"] = str(
			baked
			/ (base + "_" + channel["kind"] + (".hdr" if channel["kind"] == "emission" else ".png"))
		)
	spec = {
		"shader": shader.name,
		"output": output.name,
		"channels": channels,
		"size": size,
		"frame": bpy.context.scene.frame_current,
		"result": str(work / (base + "_result.json")),
	}
	manifest = work / (base + ".json")
	manifest.write_text(json.dumps(spec, indent=2) + "\n")
	library = work / (base + ".blend")
	bpy.data.libraries.write(str(library), {obj}, path_remap="ABSOLUTE", fake_user=True)
	command = [
		bpy.app.binary_path,
		"--background",
		"--factory-startup",
		"--disable-autoexec",
		"--threads",
		str(bpy.context.scene.render.threads),
		"--python-exit-code",
		"1",
		"--python",
		str(Path(__file__).with_name("blender_bake_worker.py")),
		"--",
		"--library",
		str(library),
		"--object",
		obj.name,
		"--uv",
		uv_name,
		"--manifest",
		str(manifest),
	]
	result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	(work / (base + ".log")).write_text(result.stdout)
	if result.returncode:
		raise RuntimeError(f"Bake worker failed for {obj.name}; see {work / (base + '.log')}")
	details = json.loads(Path(spec["result"]).read_text())
	library.unlink()
	tree = material.node_tree
	tree.nodes.clear()
	out = tree.nodes.new("ShaderNodeOutputMaterial")
	final = tree.nodes.new("ShaderNodeBsdfPrincipled")
	final.inputs["Roughness"].default_value = roughness
	final.inputs["Metallic"].default_value = metallic
	final.inputs["Transmission Weight"].default_value = transmission
	final.inputs["IOR"].default_value = ior
	for kind, entry in details["channels"].items():
		if kind == "emission" and entry["maximum"] == 0:
			continue
		image = bpy.data.images.load(entry["path"], check_existing=False)
		image.colorspace_settings.name = "Non-Color"
		tex = tree.nodes.new("ShaderNodeTexImage")
		tex.image = image
		tex.interpolation = "Linear"
		if kind == "normal":
			normal = tree.nodes.new("ShaderNodeNormalMap")
			normal.space = "TANGENT"
			normal.uv_map = uv_name
			tree.links.new(tex.outputs["Color"], normal.inputs["Color"])
			tree.links.new(normal.outputs["Normal"], final.inputs["Normal"])
		else:
			name = {
				"color": "Base Color",
				"roughness": "Roughness",
				"metallic": "Metallic",
				"alpha": "Alpha",
				"emission": "Emission Color",
			}[kind]
			if kind in ("roughness", "metallic", "alpha"):
				separate = tree.nodes.new("ShaderNodeSeparateColor")
				separate.mode = "RGB"
				tree.links.new(tex.outputs["Color"], separate.inputs[0])
				source = separate.outputs["Red"]
			else:
				source = tex.outputs["Color"]
			tree.links.new(source, final.inputs[name])
			if kind == "emission":
				final.inputs["Emission Strength"].default_value = 1
	if shader_type == "EMISSION":
		for link in list(final.inputs["Base Color"].links):
			tree.links.remove(link)
		final.inputs["Base Color"].default_value = (0, 0, 0, 1)
	tree.links.new(final.outputs[0], out.inputs["Surface"])
	# A coverage bake already resolves any source MASK threshold into coverage.
	material["xenon_alpha_mode"] = "BLEND"
	record(
		material.name,
		{
			"source": original.name,
			"selected_closure": spec["shader"],
			"roughness": roughness,
			"metallic": metallic,
			"transmission": transmission,
			"ior": ior,
			"bakes": details,
			"limitations": "Layered closure shape, sheen, coat, anisotropy and shader displacement are approximated; selected base lobe retained. Linked IOR/transmission use reported constants. View-dependent inputs are evaluated at bake directions; per-instance Object Info variation shares this mesh bake.",
		},
	)
	return material


def identity_light(light):
	light.use_nodes = True
	tree = light.node_tree
	tree.nodes.clear()
	output = tree.nodes.new("ShaderNodeOutputLight")
	emission = tree.nodes.new("ShaderNodeEmission")
	emission.inputs["Color"].default_value = (1, 1, 1, 1)
	emission.inputs["Strength"].default_value = 1
	tree.links.new(emission.outputs[0], output.inputs["Surface"])
	light.use_shadow = True


def convert_light(target, name, light, matrix, record):
	color = tuple(light.color)
	if light.use_temperature:
		color = tuple(a * b for a, b in zip(color, light.temperature_color))
	if light.type == "AREA":
		if light.cycles.is_portal:
			record(name, "Sampling-only portal omitted; no emitted radiance")
			return
		if not Exporter.identity_light_shader(light):
			record(
				name,
				"Custom area-light shader replaced by datablock energy/color/exposure/temperature",
			)
		if not light.use_shadow:
			record(name, "Shadow-disabled area lamp converted to physically shadowed emitter")
		if light.shape not in ("SQUARE", "RECTANGLE") or abs(light.spread - math.pi) > 1e-6:
			raise RuntimeError(f"{name}: conversion requires rectangular area light with spread π")
		sx = light.size
		sy = light.size_y if light.shape == "RECTANGLE" else sx
		mesh = bpy.data.meshes.new(name + "_emitter")
		mesh.from_pydata(
			[
				(-sx / 2, -sy / 2, 0),
				(-sx / 2, sy / 2, 0),
				(sx / 2, sy / 2, 0),
				(sx / 2, -sy / 2, 0),
			],
			[],
			[(0, 1, 2, 3)],
		)
		obj = bpy.data.objects.new(name + "_emitter", mesh)
		target.collection.objects.link(obj)
		obj.matrix_world = matrix
		area = (
			(matrix.to_3x3() @ Vector((sx, 0, 0)))
			.cross(matrix.to_3x3() @ Vector((0, sy, 0)))
			.length
		)
		if area <= 0:
			raise RuntimeError(f"{name}: degenerate area light")
		mat = bpy.data.materials.new(obj.name)
		mat.use_nodes = True
		tree = mat.node_tree
		tree.nodes.clear()
		output = tree.nodes.new("ShaderNodeOutputMaterial")
		emission = tree.nodes.new("ShaderNodeEmission")
		emission.inputs["Color"].default_value = (*color, 1)
		emission.inputs["Strength"].default_value = (
			light.energy * 2**light.exposure / (math.pi * area if light.normalize else math.pi)
		)
		# Cycles mesh emission is two-sided, unlike an area light. Explicitly make
		# the derived source two-sided too and record this visibility difference.
		tree.links.new(emission.outputs[0], output.inputs["Surface"])
		mesh.materials.append(mat)
		record(
			name,
			{
				"area_lamp_conversion": "Camera-visible, two-sided emissive rectangle; back-side emission and visibility differ from original area lamp",
				"normalize": light.normalize,
				"area_blender_units_squared": area,
				"emission_strength": emission.inputs["Strength"].default_value,
				"radiance_formula": "energy*2**exposure/(π*area)"
				if light.normalize
				else "energy*2**exposure/π",
			},
		)
	else:
		obj = bpy.data.objects.new(name, light)
		target.collection.objects.link(obj)
		obj.matrix_world = matrix
		if light.type == "SUN":
			light.angle = 0
			record(name, "Sun angular extent set to zero (ideal directional light)")
		elif light.type in ("POINT", "SPOT"):
			light.shadow_soft_size = 0
			light.normalize = True
			record(name, "Point/spot radius set to zero and normalized power enabled")
			if light.type == "SPOT":
				if not 0 < light.spot_size < math.pi:
					raise RuntimeError(f"{name}: spot size must be in (0,180) degrees")
				light["xenon_spot_approximation"] = True
				record(
					name,
					"Native spot uses energy/(4π), cosine smoothstep, inner half-angle=outer*(1-blend); Cycles angular profile and normalization are different",
				)
		else:
			raise RuntimeError(f"{name}: unsupported light type {light.type}")
		if not Exporter.identity_light_shader(light):
			record(
				name,
				"Custom light shader replaced by unit emission; datablock energy/color/exposure retained",
			)
		if not light.use_shadow:
			record(name, "Shadow-disabled lamp converted to physically shadowed lighting")
		identity_light(light)


def prepare(args, custom):
	root = Path(args.output).resolve()
	root.mkdir(parents=True, exist_ok=True)
	source = bpy.context.scene
	started = time.perf_counter()
	report = {
		"source": bpy.data.filepath,
		"source_frame": source.frame_current,
		"conversion": "Render-time geometry snapshot; supported materials preserved; unsupported surface graphs baked to explicit Principled approximations",
		"bake_size": custom.bake_size,
		"changes": [],
		"objects": [],
		"status": "in_progress",
	}

	def save():
		(root / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n")

	def record(where, message):
		report["changes"].append({"where": where, "message": message})
		save()

	save()
	try:
		validation_args = SimpleNamespace(**vars(args))
		validation_args.output = str(root / "validation")
		validator = Exporter(validation_args)
		if getattr(custom, "volume_proxies", False):
			from blender_prepare_volume import prepare_volumes

			report["volumes"] = prepare_volumes(source, root)
			save()
		omitted = getattr(custom, "omit_object", [])
		omit_area = getattr(custom, "omit_area_lights", False)
		captured = capture_render(source, validator, omitted, omit_area)
		if omitted or omit_area:
			record(
				"explicit omissions",
				{
					"requested_objects": omitted,
					"omit_area_lights": omit_area,
					"omitted_render_placements": captured["omitted"],
					"effect": "Selected source objects/lights are absent from the derived scene for every ray; original file unchanged",
				},
			)
		report["depsgraph_mode"] = "RENDER"
		report["source_instances"] = len(captured["instances"])
		target = bpy.data.scenes.new("Xenon comparison source")
		target.render.engine = "CYCLES"
		target.cycles.device = "CPU"
		target.cycles.samples = 1
		target.cycles.use_denoising = False
		target.frame_set(source.frame_current)
		target.unit_settings.system = source.unit_settings.system
		target.unit_settings.scale_length = source.unit_settings.scale_length
		target.render.resolution_x = args.width or round(
			source.render.resolution_x * source.render.resolution_percentage / 100
		)
		target.render.resolution_y = args.height or round(
			source.render.resolution_y * source.render.resolution_percentage / 100
		)
		target.render.resolution_percentage = 100
		target.render.threads_mode = "FIXED"
		target.render.threads = source.render.threads
		target.render.pixel_aspect_x = source.render.pixel_aspect_x
		target.render.pixel_aspect_y = source.render.pixel_aspect_y
		target.world = source.world.copy() if source.world else None
		for field in ("view_transform", "look", "exposure", "gamma"):
			try:
				setattr(target.view_settings, field, getattr(source.view_settings, field))
			except TypeError:
				pass
		if source.render.use_border:
			record(
				"scene",
				{
					"render_border_disabled": [
						source.render.border_min_x,
						source.render.border_max_x,
						source.render.border_min_y,
						source.render.border_max_y,
					],
					"crop_to_border": source.render.use_crop_to_border,
				},
			)
		compositor = (
			source.compositing_node_group is not None
			if hasattr(source, "compositing_node_group")
			else source.use_nodes
		)
		if compositor:
			record(
				"scene",
				"Authored compositor omitted; comparison source contains scene-linear render only",
			)
		if getattr(source.render, "use_motion_blur", False):
			record("scene", "Motion blur omitted; geometry frozen at the authored frame")
		if source.render.use_sequencer:
			record("scene", "Sequencer omitted; active scene evaluated directly")
		bpy.context.window.scene = target
		if custom.bake_world:
			from blender_prepare_world import prepare_world

			report["world"] = prepare_world(target, root)
			save()
		validator.scene = target
		support = {}
		parts = {}
		baked_count = 0
		preserved = set()
		simplified = set()

		def probe(material):
			before = len(validator.report["errors"])
			name = validator.material(material)
			return name, list(validator.report["errors"][before:])

		def retain(material):
			(simplified if material.get("xenon_prepared_graph", False) else preserved).add(
				material.name
			)

		instances = (
			captured["instances"][: custom.max_objects]
			if custom.max_objects
			else captured["instances"]
		)
		if len(instances) != len(captured["instances"]):
			record(
				"scene",
				f"Development limit retains {len(instances)} of {len(captured['instances'])} render instances",
			)
		placements = []
		for number, instance in enumerate(instances):
			key = instance["key"]
			mesh = captured["meshes"][key]
			if not mesh.polygons:
				continue
			if instance["visibility_changes"]:
				record(
					instance["name"], {"per_ray_visibility_reset": instance["visibility_changes"]}
				)
			if key not in parts:
				slots = list(mesh.materials)
				used = sorted({face.material_index for face in mesh.polygons})
				if any(is_volume(material) for material in slots):
					holder = bpy.data.objects.new("Volume validation", mesh)
					before = len(validator.report["errors"])
					validator.volume(holder, mesh, validator.transform @ instance["matrix"])
					bpy.data.objects.remove(holder)
					if len(validator.report["errors"]) > before:
						raise RuntimeError(json.dumps(validator.report["errors"][before:]))
					parts[key] = [mesh]
					record(
						instance["name"],
						"Homogeneous volume material preserved; exporter validates axis-aligned box boundary",
					)
				else:
					active = next(
						(uv for uv in mesh.uv_layers if uv.active_render), mesh.uv_layers.active
					)
					verdict = {}
					for slot in used:
						material = slots[slot] if slot < len(slots) else None
						matkey = material.as_pointer() if material else 0
						if matkey not in support:
							name, errors = probe(material)
							if errors and material:
								source_name = material.name
								material, changes = simplify_material(material, root)
								name, errors = probe(material)
								record(
									source_name,
									{
										"material_only_conversion": changes,
										"result": "original UVs/maps retained"
										if not errors
										else "remaining graph requires material baking",
									},
								)
							support[matkey] = (material, name, errors)
						material, name, base_errors = support[matkey]
						errors = list(base_errors)
						if name in validator.material_has_maps and active is None:
							errors.append({"message": "Texture requires active UV"})
						if any(
							active is None or active.name != uv
							for uv in validator.material_uv_maps.get(name, ())
						):
							errors.append(
								{"message": "Material requires a different active UV layer"}
							)
						verdict[slot] = (not errors, errors)
						if slot < len(slots):
							slots[slot] = material
							mesh.materials[slot] = material
					if all(verdict[slot][0] for slot in used):
						parts[key] = [mesh]
						for slot in used:
							retain(slots[slot])
					else:
						parts[key] = []
						for slot in used:
							material = slots[slot] if slot < len(slots) else None
							part = split_material(mesh, slot)
							if verdict[slot][0]:
								part.materials.append(material)
								retain(material)
							elif material is None:
								neutral = bpy.data.materials.new("Explicit unassigned neutral")
								neutral.use_nodes = True
								neutral.node_tree.nodes["Principled BSDF"].inputs[
									"Base Color"
								].default_value = (0.5, 0.5, 0.5, 1)
								part.materials.append(neutral)
								record(
									instance["name"],
									"Unassigned material explicitly replaced by neutral 0.5 reflectance",
								)
							else:
								obj = bpy.data.objects.new(
									f"{safe(instance['name'])}_{number}_slot{slot}", part
								)
								target.collection.objects.link(obj)
								obj.matrix_world = instance["matrix"]
								record(material.name, {"reason_for_conversion": verdict[slot][1]})
								bake_material(obj, material, root, custom.bake_size, record)
								baked_count += 1
								bpy.data.objects.remove(obj, do_unlink=True)
							parts[key].append(part)
			for index, part in enumerate(parts[key]):
				placements.append(
					(f"{safe(instance['name'])}_{number}_{index}", part, instance["matrix"])
				)
			report["objects"].append(
				{
					"source": instance["name"],
					"instance": instance["instance"],
					"parts": len(parts[key]),
					"polygons": len(mesh.polygons),
				}
			)
			if number % 1000 == 0:
				print(
					f"PREPARED {number + 1}/{len(instances)} render instances, {baked_count} material bakes",
					flush=True,
				)
				save()
		# Bake operators should not repeatedly evaluate tens of thousands of
		# already prepared scatter placements. Link them once all meshes are ready.
		for name, part, matrix in placements:
			obj = bpy.data.objects.new(name, part)
			target.collection.objects.link(obj)
			obj.matrix_world = matrix
		data, matrix = captured["camera"]
		camera = bpy.data.objects.new("Prepared camera", data)
		target.collection.objects.link(camera)
		camera.matrix_world = matrix
		target.camera = camera
		if data.dof.use_dof:
			if data.dof.aperture_blades or data.dof.aperture_ratio != 1:
				record(
					camera.name,
					"Non-circular depth of field converted to circular aperture with ratio 1",
				)
				data.dof.aperture_blades = 0
				data.dof.aperture_ratio = 1
			else:
				record(
					camera.name,
					"Circular depth of field, f-stop and evaluated focus distance preserved",
				)
		for name, light, matrix in captured["lights"]:
			convert_light(target, name, light, matrix, record)
		resized = resize_surface_images(target, root, getattr(custom, "max_texture_size", 0))
		if resized:
			record(
				"surface textures",
				{
					"maximum_dimension": custom.max_texture_size,
					"resized_images": resized,
					"world_images": "unchanged",
				},
			)
		report.update(
			realized_instances=len(instances),
			unique_meshes=len(parts),
			baked_material_meshes=baked_count,
			preserved_materials=sorted(preserved),
			uv_preserved_simplified_materials=sorted(simplified),
		)
		record(
			"geometry",
			"Evaluated render geometry and placements frozen at source frame; no viewport density substitution. Supported material graphs, original UVs and corner normals retained.",
		)
		bpy.context.view_layer.update()
		if bpy.data.use_autopack:
			bpy.data.use_autopack = False
			record(
				"derived source",
				"Automatic packing disabled for the export copy. Existing packed images are retained; generated images are saved alongside it. Unused missing source images are not repacked.",
			)
		bpy.ops.wm.save_as_mainfile(filepath=str(root / "comparison.blend"))
		Exporter(args).run()
		report["status"] = "ok"
		report["elapsed_seconds"] = time.perf_counter() - started
		save()
		return report
	except Exception as error:
		report["status"] = "failed"
		report["error"] = str(error)
		report["elapsed_seconds"] = time.perf_counter() - started
		save()
		scene = root / "scene.xenon"
		if scene.exists():
			scene.unlink()
		raise


def main():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--bake-size", type=int, default=256)
	parser.add_argument(
		"--bake-world",
		action="store_true",
		help="Bake the original world into a linear HDR environment (explicit approximation)",
	)
	parser.add_argument(
		"--volume-proxies",
		action="store_true",
		help="Explicitly convert supported volume graphs to reported homogeneous box proxies",
	)
	parser.add_argument(
		"--max-texture-size",
		type=int,
		default=0,
		help="Explicit cap for prepared surface textures; world images retain their resolution",
	)
	parser.add_argument(
		"--omit-area-lights",
		action="store_true",
		help="Explicitly omit area lights instead of converting them to visible emissive meshes",
	)
	parser.add_argument(
		"--omit-object",
		action="append",
		default=[],
		metavar="NAME",
		help="Explicitly omit all render placements of an exact source object name; repeat as needed",
	)
	parser.add_argument(
		"--max-objects",
		type=int,
		default=0,
		help="Development only: retain this many evaluated instances",
	)
	custom, remaining = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
	if custom.bake_size <= 0:
		parser.error("--bake-size must be positive")
	if custom.max_objects < 0:
		parser.error("--max-objects must be nonnegative")
	if custom.max_texture_size < 0:
		parser.error("--max-texture-size must be nonnegative")
	if any(not name.strip() for name in custom.omit_object):
		parser.error("--omit-object requires a nonempty name")
	prepare(arguments(remaining), custom)


if __name__ == "__main__":
	main()
