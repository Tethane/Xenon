#!/usr/bin/env python3
"""Export Blender's evaluated scene to Xenon v2 (run with blender --background).

blender --background file.blend --python scripts/blender_export.py -- --output build/export
Strict by default: inspect export-report.json for unsupported features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

AXES = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def vec(v):
	return " ".join(f"{float(x):.9g}" for x in v)


def safe(s):
	return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:100] or "unnamed"


class Exporter:
	def __init__(self, args):
		self.args = args
		self.scene = bpy.context.scene
		self.visibility = self.collection_visibility(self.scene, bpy.context.view_layer)
		self.root = Path(args.output).resolve()
		self.root.mkdir(parents=True, exist_ok=True)
		for name in ("materials", "textures", "meshes"):
			(self.root / name).mkdir(exist_ok=True)
		self.report = {
			"format": "xenon-blender-export-v2",
			"blender": bpy.app.version_string,
			"source": bpy.data.filepath,
			"frame": self.scene.frame_current,
			"errors": [],
			"approximations": [],
			"notes": [],
			"objects": [],
			"textures": [],
			"material_model": "Xenon principled subset; not numerically equivalent to Cycles Principled",
		}
		if bpy.data.filepath and Path(bpy.data.filepath).is_file():
			with Path(bpy.data.filepath).open("rb") as source:
				self.report["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
		self.materials = {}
		self.textures = {}
		self.grayscale_images = {}
		self.normal_images = {}
		self.material_uv_maps = {}
		self.material_has_maps = set()
		self.current_material = None
		self.lines = ["version 2"]
		self.unit = (
			self.scene.unit_settings.scale_length
			if self.scene.unit_settings.system != "NONE"
			else 1.0
		)
		self.transform = Matrix.Diagonal((self.unit, self.unit, self.unit, 1)) @ AXES

	def issue(self, where, problem, approximate=False):
		self.report["approximations" if approximate else "errors"].append(
			{"where": where, "message": problem}
		)

	@staticmethod
	def collection_visibility(scene, view_layer):
		"""Render-visible collection paths, independent of viewport visibility.

		Objects with multiple collection links are visible if any path permits
		them. Instance collections have their own hierarchy and do not inherit
		exclusions from an unrelated link in the scene's active view layer.
		"""
		visible = set()

		def visit_layer(layer):
			if layer.exclude or layer.collection.hide_render:
				return
			visible.update(obj.as_pointer() for obj in layer.collection.objects)
			for child in layer.children:
				visit_layer(child)

		visit_layer(view_layer.layer_collection)
		members = {}

		def instance_members(collection, visiting=None):
			key = collection.as_pointer()
			if key in members:
				return members[key]
			visiting = set() if visiting is None else set(visiting)
			if key in visiting or collection.hide_render:
				return frozenset()
			visiting.add(key)
			result = set()
			for obj in collection.objects:
				if obj.hide_render:
					continue
				result.add(obj.as_pointer())
				if obj.instance_type == "COLLECTION" and obj.instance_collection:
					result.update(instance_members(obj.instance_collection, visiting))
			for child in collection.children:
				result.update(instance_members(child, visiting))
			members[key] = frozenset(result)
			return members[key]

		instancers = {}
		for obj in scene.objects:
			if obj.instance_type == "COLLECTION" and obj.instance_collection:
				instancers[obj.as_pointer()] = instance_members(obj.instance_collection)
		return {"objects": frozenset(visible), "instancers": instancers}

	@staticmethod
	def instance_visible(instance, visibility=None):
		if not instance.show_self:
			return False
		obj = instance.object
		if not instance.is_instance:
			return not obj.hide_render and (
				visibility is None or obj.original.as_pointer() in visibility["objects"]
			)
		parent = instance.parent
		if parent and parent.hide_render:
			return False
		if visibility is not None and parent:
			parent_key = parent.original.as_pointer()
			if parent_key not in visibility["objects"]:
				return False
			members = visibility["instancers"].get(parent_key)
			if members is not None and obj.original.as_pointer() not in members:
				return False
		if not obj.hide_render:
			return True
		# A hidden particle/GN prototype can still generate visible placements.
		# Ordinary collection instances, in contrast, respect member hide_render.
		# This distinction is checked against Blender's render depsgraph in tests.
		return bool(
			instance.particle_system
			or (
				parent
				and any(
					modifier.type == "NODES" and modifier.show_render
					for modifier in parent.modifiers
				)
			)
		)

	def validate_evaluation(self, depsgraph):
		self.report["depsgraph_mode"] = depsgraph.mode
		checked = set()

		def inspect(obj):
			if obj is None or obj.original.as_pointer() in checked:
				return
			checked.add(obj.original.as_pointer())
			if (
				depsgraph.mode == "VIEWPORT"
				and obj.show_instancer_for_viewport != obj.show_instancer_for_render
			):
				self.issue(
					obj.name,
					"Instancer viewport/render visibility differs; synchronize show_instancer settings before export",
				)
			for modifier in obj.modifiers:
				if depsgraph.mode == "VIEWPORT" and modifier.show_viewport != modifier.show_render:
					self.issue(
						obj.name,
						f"Modifier {modifier.name!r} viewport/render visibility differs; apply or synchronize before export",
					)
				if not modifier.show_render:
					continue
				if modifier.type == "PARTICLE_SYSTEM":
					settings = modifier.particle_system.settings
					if settings.render_type == "NONE":
						continue
					if settings.render_type not in ("OBJECT", "COLLECTION"):
						self.issue(
							obj.name,
							f"Particle system {settings.name!r} renders {settings.render_type}; raw particle hair/paths are unsupported, convert to explicit mesh geometry before export",
						)
					if depsgraph.mode == "VIEWPORT" and settings.display_percentage != 100:
						self.issue(
							obj.name,
							f"Particle system {settings.name!r} viewport density is {settings.display_percentage}%; set display_percentage=100 to match render density",
						)
					if (
						depsgraph.mode == "VIEWPORT"
						and settings.child_type != "NONE"
						and settings.child_percent != settings.rendered_child_count
					):
						self.issue(
							obj.name,
							f"Particle system {settings.name!r} viewport/render child counts differ ({settings.child_percent}/{settings.rendered_child_count}); synchronize before export",
						)
				if (
					modifier.type == "NODES"
					and modifier.node_group
					and depsgraph.mode == "VIEWPORT"
				):
					groups = [modifier.node_group]
					visited = set()
					while groups:
						group = groups.pop()
						if group.as_pointer() in visited:
							continue
						visited.add(group.as_pointer())
						for node in group.nodes:
							if node.mute:
								continue
							if node.bl_idname == "GeometryNodeIsViewport":
								self.issue(
									obj.name,
									f"Geometry Nodes group {group.name!r} uses Is Viewport; viewport evaluation cannot establish render density/geometry, bake or remove the render-dependent branch before export",
								)
							if node.type == "GROUP" and node.node_tree:
								groups.append(node.node_tree)

		# Validate emitters even when show_self is false: their particles and
		# generated instances remain part of the scene independently of the mesh.
		for instance in depsgraph.object_instances:
			if instance.is_instance:
				if instance.parent and (
					instance.parent.hide_render
					or instance.parent.original.as_pointer() not in self.visibility["objects"]
				):
					continue
				inspect(instance.parent)
				if self.instance_visible(instance, self.visibility):
					inspect(instance.object)
			elif (
				not instance.object.hide_render
				and instance.object.original.as_pointer() in self.visibility["objects"]
			):
				inspect(instance.object)

	def image(self, node, channel="Color"):
		image = node.image
		self.material_has_maps.add(self.current_material)
		if not image:
			raise ValueError("Image Texture has no image")
		if image.source not in ("FILE", "GENERATED"):
			raise ValueError(
				f"Image source {image.source} is unsupported (UDIM/sequences require baking)"
			)
		if (
			node.projection != "FLAT"
			or node.extension != "REPEAT"
			or node.interpolation != "Linear"
		):
			raise ValueError("Only flat, repeat, linear Image Texture sampling is supported")
		if node.inputs["Vector"].is_linked:
			source = node.inputs["Vector"].links[0]
			if source.from_node.type == "UVMAP":
				if source.from_node.uv_map:
					self.material_uv_maps.setdefault(self.current_material, set()).add(
						source.from_node.uv_map
					)
			elif source.from_node.type != "TEX_COORD" or source.from_socket.name != "UV":
				raise ValueError(
					"Only the active UV coordinate source is supported; bake Mapping/generated coordinates first"
				)
		colorspace = image.colorspace_settings.name
		if colorspace not in ("sRGB", "Non-Color", "Linear", "Linear Rec.709"):
			raise ValueError(
				f"Image color space {colorspace!r} requires conversion to sRGB or linear Rec.709"
			)
		key = image.as_pointer()
		if key not in self.textures:
			name = f"{len(self.textures):03d}_{safe(image.name)}"
			source = Path(bpy.path.abspath(image.filepath, library=image.library))
			if (
				not image.packed_file
				and not image.is_dirty
				and source.is_file()
				and source.suffix.lower() in (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".hdr")
			):
				target = self.root / "textures" / (name + source.suffix.lower())
				shutil.copyfile(source, target)
			else:
				if image.is_float:
					pixels = np.empty(len(image.pixels), dtype=np.float32)
					image.pixels.foreach_get(pixels)
					if np.any(pixels < 0) or np.any(pixels > 1):
						raise ValueError(
							"Packed/generated HDR image cannot be saved to PNG without loss; save an external linear .hdr image first"
						)
				target = self.root / "textures" / (name + ".png")
				oldpath, oldformat = image.filepath_raw, image.file_format
				try:
					image.filepath_raw = str(target)
					image.file_format = "PNG"
					image.save()
				finally:
					image.filepath_raw, image.file_format = oldpath, oldformat
			self.textures[key] = target
			self.report["textures"].append(
				{
					"source": image.name,
					"path": str(target.relative_to(self.root)),
					"color_space": colorspace,
					"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
				}
			)
		return (
			self.textures[key],
			"srgb" if colorspace == "sRGB" and channel != "Alpha" else "linear",
			"a" if channel == "Alpha" else "r",
		)

	def normal_strength_image(self, texture, strength):
		if strength >= 1 or strength < 0:
			return texture
		key = (str(texture[0]), float(strength))
		if key in self.normal_images:
			return self.normal_images[key]
		source_key = next(key for key, path in self.textures.items() if path == texture[0])
		source = next(image for image in bpy.data.images if image.as_pointer() == source_key)
		if source.channels < 3:
			raise ValueError("Normal image must have RGB channels")
		width, height = source.size
		pixels = np.empty(len(source.pixels), dtype=np.float32)
		source.pixels.foreach_get(pixels)
		pixels = pixels.reshape((height, width, source.channels))[:, :, :3].copy()
		if not np.all(np.isfinite(pixels)) or np.any(pixels < 0) or np.any(pixels > 1):
			raise ValueError("Normal image values must be finite and in [0,1]")
		# Cycles scales decoded XY and blends decoded Z toward +1 below strength1.
		# Preserve that operation using the glTF XY-only normalScale convention.
		pixels[:, :, 2] = 1 - strength + strength * pixels[:, :, 2]
		values = np.rint(pixels[::-1] * 65535).astype(">u2")
		name = f"{texture[0].stem}_normal_strength_{hashlib.sha256(strength.hex().encode()).hexdigest()[:12]}.png"
		target = self.root / "textures" / name

		def chunk(kind, data):
			return (
				struct.pack(">I", len(data))
				+ kind
				+ data
				+ struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
			)

		target.write_bytes(
			b"\x89PNG\r\n\x1a\n"
			+ chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
			+ chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row.tobytes() for row in values)))
			+ chunk(b"IEND", b"")
		)
		result = (target, "linear", texture[2])
		self.normal_images[key] = result
		self.report["textures"].append(
			{
				"source": source.name,
				"path": str(target.relative_to(self.root)),
				"color_space": "linear",
				"conversion": f"Cycles tangent normal strength {strength:.9g}: blue = 1-strength+strength*blue; RGB16 PNG",
				"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
			}
		)
		return result

	def socket(self, socket, scalar=False):
		if not socket.is_linked:
			value = socket.default_value
			return (float(value) if scalar else tuple(value[:3])), None
		link = socket.links[0]
		node, output = link.from_node, link.from_socket
		if node.type == "REROUTE":
			return self.socket(node.inputs[0], scalar)
		if node.type == "RGB" and not scalar:
			return tuple(node.outputs[0].default_value[:3]), None
		if node.type == "VALUE" and scalar:
			return float(node.outputs[0].default_value), None
		if node.type == "MATH" and scalar and node.operation == "MULTIPLY" and not node.use_clamp:
			a, ta = self.socket(node.inputs[0], True)
			b, tb = self.socket(node.inputs[1], True)
			if ta and tb:
				raise ValueError("Multiplication of two textures requires baking")
			return a * b, ta or tb
		if node.type in ("MIX", "MIX_RGB") and not scalar and node.blend_type == "MULTIPLY":
			if (node.type == "MIX_RGB" and node.use_clamp) or (
				node.type == "MIX" and (node.clamp_result or node.data_type != "RGBA")
			):
				raise ValueError("Clamped or non-color Multiply requires baking")
			factor, ft = self.socket(node.inputs[0], True)
			if ft:
				raise ValueError("Textured multiply factor requires baking")
			left, right = (
				(node.inputs[6], node.inputs[7])
				if node.type == "MIX"
				else (node.inputs[1], node.inputs[2])
			)
			a, ta = self.socket(left)
			b, tb = self.socket(right)
			if ta and tb:
				raise ValueError("Multiplication of two color textures requires baking")
			if tb and factor != 1:
				raise ValueError(
					"Partial color multiply with textured right operand requires baking"
				)
			return tuple(x * ((1 - factor) + factor * y) for x, y in zip(a, b)), ta or tb
		if node.type == "TEX_IMAGE":
			if not scalar and output.name == "Alpha":
				raise ValueError(
					"Image Alpha into color requires an explicitly baked grayscale image"
				)
			texture = self.image(node, output.name)
			if scalar and texture[1] == "srgb":
				raise ValueError(
					"Scalar textures must use Non-Color/linear data, not sRGB Color output"
				)
			if scalar and output.name == "Color":
				key = node.image.as_pointer()
				if key not in self.grayscale_images:
					pixels = np.empty(len(node.image.pixels), dtype=np.float32)
					node.image.pixels.foreach_get(pixels)
					channels = node.image.channels
					pixels = pixels.reshape((-1, channels))
					self.grayscale_images[key] = channels < 3 or bool(
						np.all(np.abs(pixels[:, :3] - pixels[:, :1]) < 1e-7)
					)
				if not self.grayscale_images[key]:
					raise ValueError(
						"Colored image into scalar uses Blender luminance conversion; bake grayscale or explicitly Separate RGB channels"
					)
			return (1.0 if scalar else (1, 1, 1)), texture
		if node.type in ("SEPRGB", "SEPARATE_COLOR") and scalar:
			if node.type == "SEPARATE_COLOR" and node.mode != "RGB":
				raise ValueError("Only RGB channel separation is supported")
			incoming = node.inputs[0]
			if incoming.is_linked and incoming.links[0].from_node.type == "TEX_IMAGE":
				texture = self.image(incoming.links[0].from_node)
				if texture[1] != "linear":
					raise ValueError("Packed scalar channels require a Non-Color image")
				channel = {"R": "r", "Red": "r", "G": "g", "Green": "g", "B": "b", "Blue": "b"}[
					output.name
				]
				return 1.0, (texture[0], "linear", channel)
		raise ValueError(
			f"Unsupported input graph {node.bl_idname}.{output.name}; bake this socket to an image"
		)

	def material(self, mat):
		key = mat.as_pointer() if mat else 0
		if key in self.materials:
			return self.materials[key]
		name = f"m{len(self.materials):03d}_{safe(mat.name if mat else 'unassigned')}"
		self.materials[key] = name
		self.current_material = name
		if not mat:
			self.issue(name, "Object contains faces with no assigned material")
			return name
		data = ["version 2", f'name "{name}"', 'type "principled"']
		where = f"material {mat.name}"
		try:
			if not mat.use_nodes:
				raise ValueError(
					"Non-node materials are not exportable without explicitly authored shader semantics"
				)
			outputs = [
				n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
			]
			if len(outputs) != 1 or not outputs[0].inputs["Surface"].is_linked:
				raise ValueError("Expected one active material output with a connected surface")
			out = outputs[0]
			if out.inputs["Volume"].is_linked:
				raise ValueError("Material volume graphs are unsupported by this exporter")
			if out.inputs["Displacement"].is_linked:
				raise ValueError(
					"Shader displacement is unsupported; apply displacement as an evaluated mesh modifier"
				)
			shader = out.inputs["Surface"].links[0].from_node
			if shader.type == "REROUTE":
				shader = shader.inputs[0].links[0].from_node
			mapping = []
			if shader.type == "BSDF_PRINCIPLED":
				mapping = [
					("Base Color", "baseColor", False),
					("Metallic", "metallic", True),
					("Roughness", "roughness", True),
					("IOR", "ior", True),
					("Transmission Weight", "transmission", True),
					("Alpha", "opacity", True),
				]
				forbidden = {
					"Subsurface Weight": 0,
					"Coat Weight": 0,
					"Sheen Weight": 0,
					"Anisotropic IOR Level": 0,
					"Anisotropic Rotation": 0,
					"Specular IOR Level": 0.5,
					"Coat IOR": 1.5,
					"Thin Film Thickness": 0,
				}
				for socket_name, neutral in forbidden.items():
					s = shader.inputs.get(socket_name)
					if s and (s.is_linked or abs(s.default_value - neutral) > 1e-6):
						raise ValueError(
							f"Principled input {socket_name} is unsupported when nondefault"
						)
				specular_tint = shader.inputs.get("Specular Tint")
				if specular_tint and (
					specular_tint.is_linked
					or any(abs(v - 1) > 1e-6 for v in specular_tint.default_value[:3])
				):
					raise ValueError("Colored Specular Tint is unsupported")
				if (
					getattr(self.args, "compensated_materials", False)
					and shader.distribution == "MULTI_GGX"
				):
					transmission, transmission_texture = self.socket(
						shader.inputs["Transmission Weight"], True
					)
					ior, ior_texture = self.socket(shader.inputs["IOR"], True)
					if (
						not transmission_texture
						and transmission == 0
						and not ior_texture
						and 1 <= ior <= 3
					):
						data.append("multipleScattering 1")
						self.issue(
							where,
							"Opaque Blender MULTI_GGX mapped to Xenon isotropic GGX energy compensation; models are not numerically identical",
							approximate=True,
						)
					else:
						self.issue(
							where,
							"Blender MULTI_GGX transmission or IOR outside [1,3] retains the single-scattering approximation; energy compensation is not supported for this case",
							approximate=True,
						)
				emission_color = shader.inputs.get("Emission Color") or shader.inputs.get(
					"Emission"
				)
				emission_strength = shader.inputs.get("Emission Strength")
				if emission_color:
					color, tex = self.socket(emission_color)
					strength, strength_tex = (
						self.socket(emission_strength, True) if emission_strength else (1.0, None)
					)
					if strength_tex:
						raise ValueError(
							"Emission Strength image requires baking with emission color"
						)
					data.append("emission " + vec(v * strength for v in color))
					data.append("emissionTwoSided 1")
					if tex:
						data.append(f'emissionTexture "../textures/{tex[0].name}" {tex[1]}')
			elif shader.type == "BSDF_DIFFUSE":
				rough, tex = self.socket(shader.inputs["Roughness"], True)
				if tex or rough != 0:
					raise ValueError(
						"Cycles rough diffuse is unsupported (use zero diffuse roughness)"
					)
				data += ["ior 1", "roughness 0.5"]
				mapping = [("Color", "baseColor", False)]
			elif shader.type in ("BSDF_GLOSSY", "BSDF_ANISOTROPIC"):
				for socket_name in ("Anisotropy", "Rotation"):
					socket = shader.inputs.get(socket_name)
					if socket and (socket.is_linked or socket.default_value != 0):
						raise ValueError(f"Glossy {socket_name} must be zero")
				data += ["metallic 1"]
				mapping = [("Color", "baseColor", False), ("Roughness", "roughness", True)]
				if shader.distribution not in ("GGX", "MULTI_GGX"):
					raise ValueError("Only GGX glossy distribution is supported")
				if shader.distribution == "MULTI_GGX":
					if not getattr(self.args, "compensated_materials", False):
						raise ValueError(
							"MULTI_GGX requires explicit --compensated-materials approximation"
						)
					data.append("multipleScattering 1")
					self.issue(
						where,
						"Blender MULTI_GGX Glossy mapped to Xenon isotropic GGX energy compensation; models are not numerically identical",
						approximate=True,
					)
			elif shader.type == "BSDF_GLASS":
				data += ["transmission 1"]
				mapping = [
					("Color", "baseColor", False),
					("Roughness", "roughness", True),
					("IOR", "ior", True),
				]
				if shader.distribution != "GGX":
					raise ValueError("Only GGX glass distribution is supported")
			elif shader.type == "EMISSION":
				color, tex = self.socket(shader.inputs["Color"])
				strength, strength_tex = self.socket(shader.inputs["Strength"], True)
				if strength_tex:
					raise ValueError("Textured emission strength requires baking")
				data += [
					"baseColor 0 0 0",
					"ior 1",
					"emission " + vec(v * strength for v in color),
					"emissionTwoSided 1",
				]
				if tex:
					data.append(f'emissionTexture "../textures/{tex[0].name}" {tex[1]}')
			else:
				raise ValueError(f"Unsupported surface shader {shader.bl_idname}")
			for input_name, key_name, scalar in mapping:
				s = shader.inputs.get(input_name)
				if not s:
					continue
				if key_name == "opacity" and mat.get("xenon_alpha_mode") == "MASK" and s.is_linked:
					# Blender's glTF importer expresses MASK as 1-(alpha<cutoff).
					# Preserve the source cutoff while exporting its pre-threshold input.
					threshold = s.links[0].from_node
					if (
						threshold.type == "MATH"
						and threshold.operation == "SUBTRACT"
						and not threshold.use_clamp
						and not threshold.inputs[0].is_linked
						and threshold.inputs[0].default_value == 1
						and threshold.inputs[1].is_linked
					):
						less = threshold.inputs[1].links[0].from_node
						cutoff = float(mat.get("xenon_alpha_cutoff", 0.5))
						if (
							less.type == "MATH"
							and less.operation == "LESS_THAN"
							and not less.use_clamp
							and not less.inputs[1].is_linked
							and abs(less.inputs[1].default_value - cutoff) < 1e-7
						):
							s = less.inputs[0]
				value, texture = self.socket(s, scalar)
				if texture and key_name not in ("baseColor", "roughness", "metallic", "opacity"):
					raise ValueError(
						f"Textured {key_name} is unsupported; only a constant can be exported"
					)
				if key_name == "opacity" and (texture or value != 1):
					mode = mat.get("xenon_alpha_mode", "BLEND")
					if mode == "MASK":
						cutoff = float(mat.get("xenon_alpha_cutoff", 0.5))
						if cutoff > 1 and not texture and value == 0:
							cutoff = 0.5  # glTF importer already resolved this mask to fully transparent.
						if not 0 <= cutoff <= 1:
							raise ValueError("Unresolved mask cutoff outside [0,1] requires baking")
						data += ["opacityMode mask", f"alphaCutoff {cutoff:.9g}"]
					elif mode == "BLEND":
						data.append("opacityMode blend")
					else:
						raise ValueError(f"Unknown opacity source mode {mode!r}")
				data.append(key_name + " " + (f"{value:.9g}" if scalar else vec(value)))
				if texture:
					data.append(
						f'{key_name}Texture "../textures/{texture[0].name}" {texture[1]}'
						+ (f" {texture[2]}" if scalar else "")
					)
			normal = shader.inputs.get("Normal")
			if normal and normal.is_linked:
				node = normal.links[0].from_node
				if node.type == "NORMAL_MAP" and node.space == "TANGENT":
					if node.uv_map:
						self.material_uv_maps.setdefault(name, set()).add(node.uv_map)
					color_scale, texture = self.socket(node.inputs["Color"])
					strength, tex_strength = self.socket(node.inputs["Strength"], True)
					if (
						not texture
						or texture[1] != "linear"
						or tex_strength
						or any(abs(v - 1) > 1e-7 for v in color_scale)
					):
						raise ValueError(
							"Tangent normal map requires a linear image and constant strength"
						)
					if strength < 0:
						raise ValueError("Normal map strength must be nonnegative")
					texture = self.normal_strength_image(texture, strength)
					data += [
						f'normalTexture "../textures/{texture[0].name}" linear',
						f"normalScale {strength:.9g}",
					]
				elif node.type == "BUMP" and not node.inputs["Normal"].is_linked:
					height_scale, texture = self.socket(node.inputs["Height"], True)
					strength, st = self.socket(node.inputs["Strength"], True)
					distance, dt = self.socket(node.inputs["Distance"], True)
					if not texture or st or dt:
						raise ValueError(
							"Bump requires a linear image, constant strength and constant distance"
						)
					data += [
						f'bumpTexture "../textures/{texture[0].name}" linear {texture[2]}',
						f"bumpScale {height_scale * strength * distance * (-1 if node.invert else 1):.9g}",
					]
					self.report["notes"].append(
						f"{mat.name}: bump derivative scaling is a material-model difference from Cycles"
					)
				else:
					raise ValueError(
						"Normal input must be a tangent Normal Map or standalone image Bump node"
					)
		except (ValueError, KeyError) as error:
			self.issue(where, str(error))
		if "emissionTwoSided 1" in data:
			required = self.report.setdefault("required_features", [])
			if "emissionTwoSided" not in required:
				required.append("emissionTwoSided")
		(self.root / "materials" / (name + ".mat")).write_text("\n".join(data) + "\n")
		self.lines.append(f'matfile "materials/{name}.mat"')
		return name

	def camera(self):
		scene = self.scene
		camera = scene.camera
		if not camera or camera.data.type != "PERSP":
			self.issue("camera", "Active perspective camera required")
			return
		c = camera.data
		if c.shift_x != 0 or c.shift_y != 0:
			self.issue("camera", "Camera lens shift is unsupported")
		if scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
			self.issue("camera", "Nonsquare pixel aspect is unsupported")
		matrix = self.transform @ camera.matrix_world
		eye = matrix.translation
		direction = -(matrix.to_3x3() @ Vector((0, 0, 1))).normalized()
		up = (matrix.to_3x3() @ Vector((0, 1, 0))).normalized()
		width = self.args.width or round(
			scene.render.resolution_x * scene.render.resolution_percentage / 100
		)
		height = self.args.height or round(
			scene.render.resolution_y * scene.render.resolution_percentage / 100
		)
		# view_frame incorporates sensor fit, aspect ratio and camera lens exactly.
		old = (
			scene.render.resolution_x,
			scene.render.resolution_y,
			scene.render.resolution_percentage,
		)
		scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = (
			width,
			height,
			100,
		)
		frame = c.view_frame(scene=scene)
		fov = math.degrees(2 * math.atan(max(abs(v.y / v.z) for v in frame)))
		scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = (
			old
		)
		self.lines += [
			f"config {width} {height} {self.args.samples}",
			f"camera {vec(eye)} {vec(eye + direction)} {fov:.9g}",
			f"camera_up {vec(up)}",
		]
		if c.dof.use_dof:
			distance = c.dof.focus_distance * self.unit
			if c.dof.focus_object:
				distance = (
					abs(
						(
							c.dof.focus_object.matrix_world.translation
							- camera.matrix_world.translation
						).dot(-camera.matrix_world.col[2].to_3d().normalized())
					)
					* self.unit
				)
			self.lines.append(
				f"lens {c.lens / 1000 / c.dof.aperture_fstop * self.unit:.9g} {distance:.9g}"
			)
			if c.dof.aperture_blades or c.dof.aperture_ratio != 1:
				self.issue("camera", "Only circular aperture with ratio 1 is supported")

	def world(self):
		world = self.scene.world
		if not world:
			self.lines.append("sky 0 0 0 0 0 0 0 0 0 1 1")
			return
		try:
			if not world.use_nodes:
				color = tuple(world.color)
			else:
				output = next(
					n
					for n in world.node_tree.nodes
					if n.type == "OUTPUT_WORLD" and n.is_active_output
				)
				if output.inputs["Volume"].is_linked:
					raise ValueError("World volumes are not yet supported by this exporter")
				shader = output.inputs["Surface"].links[0].from_node
				if shader.type != "BACKGROUND":
					raise ValueError("World surface must be a Background shader")
				color_socket = shader.inputs["Color"]
				if (
					color_socket.is_linked
					and color_socket.links[0].from_node.type == "TEX_ENVIRONMENT"
				):
					node = color_socket.links[0].from_node
					if (
						node.projection != "EQUIRECTANGULAR"
						or node.interpolation != "Linear"
						or node.inputs["Vector"].is_linked
					):
						raise ValueError(
							"Environment map requires equirectangular, linear sampling, and unlinked Vector"
						)
					image = node.image
					if image is None or image.colorspace_settings.name not in (
						"Non-Color",
						"Linear",
						"Linear Rec.709",
					):
						raise ValueError("Environment image must be scene-linear Rec.709")
					strength, st = self.socket(shader.inputs["Strength"], True)
					if st:
						raise ValueError("Environment strength must be constant")
					source = Path(bpy.path.abspath(image.filepath, library=image.library))
					target = self.root / "textures" / "environment.hdr"
					if (
						not image.packed_file
						and not image.is_dirty
						and source.is_file()
						and source.suffix.lower() == ".hdr"
					):
						shutil.copyfile(source, target)
					else:
						oldpath, oldformat = image.filepath_raw, image.file_format
						try:
							image.filepath_raw = str(target)
							image.file_format = "HDR"
							image.save()
						finally:
							image.filepath_raw, image.file_format = oldpath, oldformat
					self.lines.append(f'environment "textures/environment.hdr" {strength:.9g} 0')
					self.report["textures"].append(
						{
							"source": image.name,
							"path": "textures/environment.hdr",
							"color_space": "linear",
							"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
						}
					)
					# Blender u=.5-atan2(y,x)/(2pi), v is bottom-up. Axis
					# conversion (x,z,-y) makes Xenon's top-down mapping agree.
					return
				color, texture = self.socket(color_socket)
				strength, st = self.socket(shader.inputs["Strength"], True)
				if texture or st:
					raise ValueError(
						"Environment maps require the environment-map contract; procedural worlds must be baked"
					)
				color = tuple(v * strength for v in color)
			self.lines.append("sky " + " ".join([vec(color)] * 3) + " 1 1")
		except (ValueError, KeyError, StopIteration, IndexError) as error:
			self.issue("world", str(error))

	def lights(self, stream, vertex):
		for instance in bpy.context.evaluated_depsgraph_get().object_instances:
			obj = instance.object
			if obj.type != "LIGHT" or not self.instance_visible(instance, self.visibility):
				continue
			matrix = instance.matrix_world
			light = obj.data
			if light.type == "SUN":
				if not self.identity_light_shader(light) or not light.use_shadow:
					self.issue(obj.name, "Sun requires shadows and no custom light shader")
					continue
				if light.angle > 1e-7:
					self.issue(
						obj.name,
						"Finite sun angular extent unsupported; set angle=0 for ideal directional source",
					)
					continue
				d = (self.transform.to_3x3() @ matrix.to_3x3() @ Vector((0, 0, 1))).normalized()
				color = Vector(light.color)
				if light.use_temperature:
					color = Vector(tuple(a * b for a, b in zip(color, light.temperature_color)))
				self.lines.append(
					f"sunlight {vec(d)} {vec(color)} {light.energy * 2**light.exposure:.9g}"
				)
			elif light.type in ("POINT", "SPOT"):
				if light.type == "SPOT" and not light.get("xenon_spot_approximation", False):
					self.issue(
						obj.name,
						"Spot angular profile requires explicit prepared-source approximation",
					)
					continue
				if light.shadow_soft_size > 1e-7:
					self.issue(
						obj.name,
						"Finite radius point light unsupported; use radius 0 or emissive geometry",
					)
					continue
				if (
					not light.normalize
					or not light.use_shadow
					or not self.identity_light_shader(light)
				):
					self.issue(
						obj.name,
						"Point lights require normalized power, shadows, and no custom light shader",
					)
					continue
				color = Vector(light.color)
				if light.use_temperature:
					color = Vector(tuple(a * b for a, b in zip(color, light.temperature_color)))
				# Cycles uses Blender-unit distances for inverse-square lighting.
				# Scaling geometry to meters requires intensity to scale by unit².
				intensity = color * (
					light.energy * 2**light.exposure * self.unit**2 / (4 * math.pi)
				)
				position = self.transform @ matrix.translation
				if light.type == "SPOT":
					outer = math.degrees(light.spot_size) * 0.5
					if not 0 < outer < 90:
						self.issue(obj.name, "Prepared spot half-angle must be in (0,90) degrees")
						continue
					inner = outer * (1 - light.spot_blend)
					direction = (
						self.transform.to_3x3() @ matrix.to_3x3() @ Vector((0, 0, -1))
					).normalized()
					self.lines.append(
						f"spotlight {vec(position)} {vec(direction)} {vec(intensity)} {inner:.9g} {outer:.9g}"
					)
					self.issue(
						obj.name,
						"Prepared spot approximation uses energy/(4π), cosine smoothstep, and inner half-angle=outer*(1-blend); Cycles angular profile and flux normalization differ",
						approximate=True,
					)
				else:
					self.lines.append(f"pointlight {vec(position)} {vec(intensity)}")
			elif light.type == "AREA" and light.shape in ("SQUARE", "RECTANGLE"):
				self.issue(
					obj.name,
					"Blender area light power/visibility semantics differ; author an emissive mesh for exact export",
				)
			else:
				self.issue(
					obj.name,
					f"{light.type} light unsupported; author emissive geometry or an ideal sun",
				)
		return vertex

	@staticmethod
	def identity_light_shader(light):
		if not light.use_nodes:
			return True
		# Since Blender 5.2, even a factory light has use_nodes=True. Cycles
		# multiplies its shader by the light data's color/energy/exposure. Only
		# a constant unit emission graph leaves those exported values unchanged.
		tree = light.node_tree
		if tree is None:
			return False
		outputs = [
			node for node in tree.nodes if node.type == "OUTPUT_LIGHT" and node.is_active_output
		]
		if len(outputs) != 1 or outputs[0].mute:
			return False
		surface = outputs[0].inputs.get("Surface")
		if surface is None or len(surface.links) != 1:
			return False
		emission = surface.links[0].from_node
		return (
			emission.type == "EMISSION"
			and not emission.mute
			and not any(socket.is_linked for socket in emission.inputs)
			and tuple(emission.inputs["Color"].default_value) == (1.0, 1.0, 1.0, 1.0)
			and emission.inputs["Strength"].default_value == 1.0
		)

	def volume(self, obj, mesh, matrix):
		volume_materials = []
		for slot in obj.material_slots:
			mat = slot.material
			if mat and mat.use_nodes:
				output = next(
					(
						n
						for n in mat.node_tree.nodes
						if n.type == "OUTPUT_MATERIAL" and n.is_active_output
					),
					None,
				)
				if output and output.inputs["Volume"].is_linked:
					volume_materials.append((mat, output))
		if not volume_materials:
			return False
		try:
			if len(volume_materials) != 1 or len(obj.material_slots) != 1:
				raise ValueError("Volume box requires one material on every face")
			mat, output = volume_materials[0]
			if output.inputs["Surface"].is_linked or output.inputs["Displacement"].is_linked:
				raise ValueError("Volume box must have no surface/displacement closure")
			if abs(self.unit - 1) > 1e-8:
				raise ValueError("Volume export currently requires one Blender unit per meter")
			vertices = [matrix @ vertex.co for vertex in mesh.vertices]
			lower = Vector(tuple(min(v[a] for v in vertices) for a in range(3)))
			upper = Vector(tuple(max(v[a] for v in vertices) for a in range(3)))
			tolerance = max((upper - lower).length, 1) * 1e-6
			if len(vertices) != 8 or min(upper - lower) <= tolerance:
				raise ValueError(
					"Homogeneous volume boundary must be an eight-corner nondegenerate box"
				)
			corner_codes = []
			for vertex in vertices:
				code = []
				for axis in range(3):
					if abs(vertex[axis] - lower[axis]) < tolerance:
						code.append(0)
					elif abs(vertex[axis] - upper[axis]) < tolerance:
						code.append(1)
					else:
						raise ValueError(
							"Rotated/sheared volume box is not axis aligned in Xenon coordinates"
						)
				corner_codes.append(tuple(code))
			if len(set(corner_codes)) != 8:
				raise ValueError("Volume boundary does not have all eight box corners")
			edge_counts = {}
			for face in mesh.polygons:
				codes = [corner_codes[index] for index in face.vertices]
				if not any(len({code[axis] for code in codes}) == 1 for axis in range(3)):
					raise ValueError("Volume face is not on the box boundary")
				indices = list(face.vertices)
				for a, b in zip(indices, indices[1:] + indices[:1]):
					edge = tuple(sorted((a, b)))
					edge_counts[edge] = edge_counts.get(edge, 0) + 1
			if any(count != 2 for count in edge_counts.values()):
				raise ValueError("Volume box must have a closed manifold boundary")
			sigma_a = Vector((0, 0, 0))
			sigma_s = Vector((0, 0, 0))
			anisotropies = []

			def visit(node):
				if node.type == "ADD_SHADER":
					for socket in node.inputs:
						if socket.is_linked:
							visit(socket.links[0].from_node)
					return
				if node.type not in ("VOLUME_ABSORPTION", "VOLUME_SCATTER"):
					raise ValueError(f"Unsupported volume closure {node.bl_idname}")
				color, ct = self.socket(node.inputs["Color"])
				density, dt = self.socket(node.inputs["Density"], True)
				if ct or dt:
					raise ValueError("Only homogeneous constant volume coefficients are supported")
				if density < 0 or any(c < 0 or c > 1 for c in color):
					raise ValueError("Volume colors must be in [0,1] and density nonnegative")
				if node.type == "VOLUME_ABSORPTION":
					for axis in range(3):
						sigma_a[axis] += (1 - color[axis]) * density
				else:
					if getattr(node, "phase", "HENYEY_GREENSTEIN") != "HENYEY_GREENSTEIN":
						raise ValueError("Only Henyey-Greenstein volume scattering is supported")
					g, gt = self.socket(node.inputs["Anisotropy"], True)
					if gt or not -1 < g < 1:
						raise ValueError("Volume anisotropy must be constant in (-1,1)")
					anisotropies.append(g)
					for axis in range(3):
						sigma_s[axis] += color[axis] * density

			visit(output.inputs["Volume"].links[0].from_node)
			if anisotropies and max(anisotropies) - min(anisotropies) > 1e-7:
				raise ValueError("Added scatter closures must use the same anisotropy")
			g = anisotropies[0] if anisotropies else 0
			self.lines.append(
				f"volume_box {vec(lower)} {vec(upper)} {vec(sigma_a)} {vec(sigma_s)} {g:.9g}"
			)
			self.report["objects"].append(
				{
					"name": obj.name,
					"type": "homogeneous_volume_box",
					"sigma_a": list(sigma_a),
					"sigma_s": list(sigma_s),
					"g": g,
				}
			)
		except (ValueError, KeyError) as error:
			self.issue(obj.name, str(error))
		return True

	def mesh_data(self, obj, mesh):
		active = next(
			(layer for layer in mesh.uv_layers if layer.active_render), mesh.uv_layers.active
		)
		used_slots = {tri.material_index for tri in mesh.loop_triangles}
		materials = [
			self.material(slot.material) if index in used_slots else None
			for index, slot in enumerate(obj.material_slots)
		]
		for material_name in materials:
			if material_name in self.material_has_maps and active is None:
				self.issue(
					obj.name,
					f"Material {material_name} has maps but mesh has no active render UV layer",
				)
			for required_uv in self.material_uv_maps.get(material_name, ()):
				if active is None or active.name != required_uv:
					self.issue(
						obj.name,
						f"Material {material_name} requires UV layer {required_uv!r}; active render layer is {active.name if active else None!r}",
					)
		if not materials:
			materials = [self.material(None)]
		positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
		mesh.vertices.foreach_get("co", positions.ravel())
		normals = np.empty((len(mesh.loops), 3), dtype=np.float32)
		mesh.corner_normals.foreach_get("vector", normals.ravel())
		corners = np.empty(len(mesh.loops), dtype=np.int32)
		mesh.loops.foreach_get("vertex_index", corners)
		uv = np.zeros((len(mesh.loops), 2), dtype=np.float32)
		if active:
			active.data.foreach_get("uv", uv.ravel())
		triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
		mesh.loop_triangles.foreach_get("loops", triangles.ravel())
		slots = np.empty(len(mesh.loop_triangles), dtype=np.int32)
		mesh.loop_triangles.foreach_get("material_index", slots)
		if slots.size and (slots.min() < 0 or slots.max() >= len(materials)):
			raise ValueError("Mesh has an invalid material slot index")
		return (
			positions,
			normals,
			corners,
			uv,
			triangles,
			slots,
			materials,
			active.name if active else None,
		)

	def write_mesh(self, stream, data, matrix, vertex=1):
		positions, normals, corners, uv, triangles, slots, materials, _ = data
		positions = positions @ np.asarray(matrix.to_3x3()).T + np.asarray(matrix.translation)
		normals = normals @ np.asarray(matrix.to_3x3().inverted().transposed()).T
		normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
		mirrored = matrix.to_3x3().determinant() < 0
		current_material = -1
		for triangle, slot in zip(triangles, slots):
			if slot != current_material:
				current_material = slot
				stream.write(f"usemtl {materials[slot]}\n")
			if mirrored:
				triangle = triangle[::-1]
			for loop in triangle:
				stream.write(
					f"v {vec(positions[corners[loop]])}\nvt {vec(uv[loop])}\nvn {vec(normals[loop])}\n"
				)
			stream.write("f " + " ".join(f"{k}/{k}/{k}" for k in range(vertex, vertex + 3)) + "\n")
			vertex += 3
		return vertex

	def geometry(self):
		depsgraph = bpy.context.evaluated_depsgraph_get()
		self.validate_evaluation(depsgraph)
		native = getattr(self.args, "instances", False)
		vertex = 1
		faces = 0
		unique_faces = 0
		assets = {}
		placements = []
		placement_cache = {}
		mesh_evaluations = 0
		stream = None if native else (self.root / "meshes" / "scene.obj").open("w")
		if stream:
			stream.write("# Xenon Blender export; world-space meters, Y up, right handed\n")
		try:
			for instance in depsgraph.object_instances:
				obj = instance.object
				if not self.instance_visible(instance, self.visibility) or obj.type in (
					"CAMERA",
					"LIGHT",
					"EMPTY",
					"ARMATURE",
				):
					continue
				visibility = (
					"visible_camera",
					"visible_diffuse",
					"visible_glossy",
					"visible_transmission",
					"visible_volume_scatter",
					"visible_shadow",
				)
				if (
					any(not getattr(obj, field, True) for field in visibility)
					or obj.is_holdout
					or obj.is_shadow_catcher
				):
					self.issue(
						obj.name,
						"Per-ray visibility, holdouts, and shadow catchers require unsupported transport semantics",
					)
				if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
					self.issue(obj.name, f"Unsupported object type {obj.type}")
					continue
				matrix = self.transform @ instance.matrix_world
				if abs(matrix.to_3x3().determinant()) < 1e-20:
					self.issue(obj.name, "Singular object transform is unsupported")
					continue
				# Reuse an evaluated mesh only when it has no object deformation.
				# Material overrides remain part of the key. Different evaluated
				# meshes still pass through the exact content hash below.
				cache_key = None
				if (
					native
					and obj.type == "MESH"
					and not obj.original.modifiers
					and not obj.original.data.shape_keys
				):
					cache_key = (
						obj.data.as_pointer(),
						tuple(
							(slot.link, slot.material.as_pointer() if slot.material else 0)
							for slot in obj.material_slots
						),
					)
				if cache_key in placement_cache:
					asset_index, triangle_count, uv_layer = placement_cache[cache_key]
					placements.append(
						f"instance {asset_index} "
						+ vec(matrix[row][column] for row in range(3) for column in range(4))
					)
					faces += triangle_count
					self.report["objects"].append(
						{
							"name": obj.name,
							"triangles": triangle_count,
							"instance": instance.is_instance,
							"uv_layer": uv_layer,
							"mesh_asset": asset_index,
						}
					)
					continue
				mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
				if mesh is None:
					self.issue(obj.name, "Evaluated mesh unavailable")
					continue
				try:
					mesh.calc_loop_triangles()
					matrix = self.transform @ instance.matrix_world
					if self.volume(obj, mesh, matrix) or not mesh.loop_triangles:
						continue
					if abs(matrix.to_3x3().determinant()) < 1e-20:
						raise ValueError("Singular object transform is unsupported")
					data = self.mesh_data(obj, mesh)
					mesh_evaluations += 1
					if native:
						# Hash evaluated geometry plus material binding, never source data
						# identity alone: modifiers and object-linked materials can differ.
						digest = hashlib.sha256()
						for array in data[:6]:
							digest.update(str(array.shape).encode())
							digest.update(array.tobytes())
						digest.update(json.dumps(data[6]).encode())
						key = digest.hexdigest()
						if key not in assets:
							index = len(assets)
							assets[key] = index
							path = f"meshes/asset_{index:04d}.obj"
							with (self.root / path).open("w") as asset:
								asset.write(
									"# Evaluated local-space mesh; placement applies units and axis conversion\n"
								)
								self.write_mesh(asset, data, Matrix.Identity(4))
							self.lines.append(f'mesh "{path}"')
							unique_faces += len(mesh.loop_triangles)
						placements.append(
							f"instance {assets[key]} "
							+ vec(matrix[row][column] for row in range(3) for column in range(4))
						)
						if cache_key is not None:
							placement_cache[cache_key] = (
								assets[key],
								len(mesh.loop_triangles),
								data[7],
							)
					else:
						stream.write(f"o {safe(obj.name)}_{len(self.report['objects'])}\n")
						vertex = self.write_mesh(stream, data, matrix, vertex)
					faces += len(mesh.loop_triangles)
					self.report["objects"].append(
						{
							"name": obj.name,
							"triangles": len(mesh.loop_triangles),
							"instance": instance.is_instance,
							"uv_layer": data[7],
							**({"mesh_asset": assets[key]} if native else {}),
						}
					)
				except ValueError as error:
					self.issue(obj.name, str(error))
				finally:
					obj.to_mesh_clear()
		finally:
			if stream:
				stream.close()
		self.lights(None, vertex)
		self.report["triangles"] = faces
		if native:
			self.lines.append("instances_explicit")
			self.lines.extend(placements)
			self.report.update(
				mesh_assets=len(assets),
				mesh_placements=len(placements),
				unique_triangles=unique_faces,
				mesh_data_evaluations=mesh_evaluations,
			)
			self.report["notes"].append(
				"Exact evaluated geometry and material bindings are deduplicated; native affine instances retain shared mesh storage"
			)
		else:
			self.report["notes"].append(
				"Evaluated instances are realized into world-space triangles; use --instances for shared mesh storage"
			)
			if faces:
				self.lines.append('mesh "meshes/scene.obj"')

	def run(self):
		start = time.perf_counter()
		if self.scene.render.use_border:
			self.issue("scene", "Render borders/cropping are unsupported; disable before export")
		if getattr(self.scene.render, "use_motion_blur", False):
			self.issue("scene", "Motion blur is unsupported; disable before export")
		# Blender 5.2 keeps deprecated use_nodes=True even without a compositor.
		compositing = (
			self.scene.compositing_node_group is not None
			if hasattr(self.scene, "compositing_node_group")
			else self.scene.use_nodes
		)
		if compositing:
			self.issue("scene", "Compositing nodes must be disabled for a shared linear reference")
		self.camera()
		self.world()
		self.geometry()
		self.report["export_seconds"] = time.perf_counter() - start
		self.report["status"] = "unsupported" if self.report["errors"] else "ok"
		(self.root / "export-report.json").write_text(json.dumps(self.report, indent=2) + "\n")
		scene_path = self.root / "scene.xenon"
		if self.report["errors"]:
			if scene_path.exists():
				scene_path.unlink()
			print(
				json.dumps(
					{
						"status": "unsupported",
						"report": str(self.root / "export-report.json"),
						"errors": self.report["errors"],
					},
					indent=2,
				)
			)
			raise RuntimeError(
				f"Export rejected {len(self.report['errors'])} unsupported features; see export-report.json"
			)
		scene_path.write_text("\n".join(self.lines) + "\n")
		print(
			f"EXPORTED {scene_path}: {self.report['triangles']} triangles, {len(self.materials)} materials"
		)
		if self.args.reference:
			render_reference(self.scene, self.args, self.root)


def render_reference(scene, args, root):
	scene.render.engine = "CYCLES"
	scene.cycles.samples = args.reference_samples or args.samples
	scene.cycles.seed = args.seed
	scene.cycles.use_denoising = False
	scene.cycles.use_adaptive_sampling = False
	# Xenon samples uniformly within each pixel; use the same reconstruction
	# support instead of Cycles' wider default Blackman-Harris filter.
	scene.cycles.pixel_filter_type = "BOX"
	scene.cycles.max_bounces = args.bounces
	scene.cycles.diffuse_bounces = args.bounces
	scene.cycles.glossy_bounces = args.bounces
	scene.cycles.transmission_bounces = args.bounces
	scene.cycles.volume_bounces = args.bounces
	scene.cycles.transparent_max_bounces = args.bounces
	scene.cycles.sample_clamp_direct = 0
	scene.cycles.sample_clamp_indirect = 0
	scene.cycles.use_light_tree = True
	# Match Exporter.camera(): overrides select final pixel dimensions, while
	# authored percentage applies to each unspecified dimension.
	percentage = scene.render.resolution_percentage
	width = args.width or round(scene.render.resolution_x * percentage / 100)
	height = args.height or round(scene.render.resolution_y * percentage / 100)
	scene.render.resolution_x, scene.render.resolution_y = width, height
	scene.render.resolution_percentage = 100
	scene.render.image_settings.file_format = "OPEN_EXR"
	scene.render.image_settings.color_mode = "RGB"
	scene.render.image_settings.color_depth = "32"
	scene.render.filepath = str(root / "cycles.exr")
	if args.cycles_device != "CPU":
		prefs = bpy.context.preferences.addons["cycles"].preferences
		prefs.compute_device_type = args.cycles_device
		prefs.get_devices()
		enabled = []
		for device in prefs.devices:
			device.use = device.type == args.cycles_device
			if device.use:
				enabled.append(device.name)
		if not enabled:
			raise RuntimeError(
				f"No {args.cycles_device} devices available (no silent CPU fallback)"
			)
		scene.cycles.device = "GPU"
	else:
		scene.cycles.device = "CPU"
		enabled = ["CPU"]
	start = time.perf_counter()
	bpy.ops.render.render(write_still=True)
	seconds = time.perf_counter() - start
	# Reload EXR: Render Result pixel access is empty in some background Blender builds.
	image = bpy.data.images.load(str(root / "cycles.exr"), check_existing=False)
	width, height = image.size
	pixels = list(image.pixels[:])
	with (root / "cycles.pfm").open("wb") as stream:
		stream.write(f"PF\n{width} {height}\n-1.0\n".encode())
		for y in range(height):
			row = []
			for x in range(width):
				row.extend(pixels[4 * (y * width + x) : 4 * (y * width + x) + 3])
			stream.write(struct.pack("<" + "f" * len(row), *row))
	metadata = {
		"renderer": "Blender Cycles",
		"blender": bpy.app.version_string,
		"devices": enabled,
		"width": width,
		"height": height,
		"samples": scene.cycles.samples,
		"seed": args.seed,
		"max_bounces": args.bounces,
		"render_seconds": seconds,
		"denoising": False,
		"adaptive_sampling": False,
		"clamping": False,
		"pixel_filter": "BOX",
		"linear_color_space": "scene-linear Rec.709",
		"source": bpy.data.filepath,
	}
	(root / "cycles-metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
	bpy.data.images.remove(image)


def arguments(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", required=True)
	parser.add_argument("--width", type=int)
	parser.add_argument("--height", type=int)
	parser.add_argument("--samples", type=int, default=64)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--bounces", type=int, default=8)
	parser.add_argument(
		"--instances",
		action="store_true",
		help="Deduplicate evaluated meshes and export native affine placements",
	)
	parser.add_argument(
		"--compensated-materials",
		action="store_true",
		help="Map eligible opaque Blender MULTI_GGX to the documented Xenon energy-compensated approximation",
	)
	parser.add_argument("--reference", action="store_true")
	parser.add_argument(
		"--reference-only",
		action="store_true",
		help="Render Cycles from the open .blend without re-exporting geometry",
	)
	parser.add_argument("--reference-samples", type=int)
	parser.add_argument("--cycles-device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
	return parser.parse_args(argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :])


if __name__ == "__main__":
	args = arguments()
	if args.reference_only:
		root = Path(args.output).resolve()
		root.mkdir(parents=True, exist_ok=True)
		render_reference(bpy.context.scene, args, root)
	else:
		Exporter(args).run()
