#!/usr/bin/env python3
"""Isolated CPU baking of one prepared material; all outputs are scene-linear."""

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path
import bpy
import numpy as np


def png16(path, pixels):
	height, width, _ = pixels.shape
	values = np.rint(np.clip(pixels[::-1, :, :3], 0, 1) * 65535).astype(">u2")

	def chunk(kind, data):
		return (
			struct.pack(">I", len(data))
			+ kind
			+ data
			+ struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
		)

	Path(path).write_bytes(
		b"\x89PNG\r\n\x1a\n"
		+ chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
		+ chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row.tobytes() for row in values)))
		+ chunk(b"IEND", b"")
	)


def coverage(tree, socket, notes, seen=None):
	"""Build RGB-neutral coverage for transparent/mix/Principled surface graphs."""
	if not socket.is_linked:
		return 0.0
	node = socket.links[0].from_node
	seen = set() if seen is None else set(seen)
	if node.as_pointer() in seen:
		raise RuntimeError("Cyclic surface coverage graph")
	seen.add(node.as_pointer())
	if node.type == "REROUTE":
		return coverage(tree, node.inputs[0], notes, seen)
	if node.type == "BSDF_TRANSPARENT":
		notes.append(
			"Transparent BSDF tint is reduced to scalar coverage (colored transmission through transparent closures is not retained)"
		)
		return 0.0
	if node.type == "BSDF_PRINCIPLED":
		value = node.inputs["Alpha"]
		if not value.is_linked:
			return max(0, min(1, float(value.default_value)))
		clamp = tree.nodes.new("ShaderNodeClamp")
		tree.links.new(value.links[0].from_socket, clamp.inputs["Value"])
		return clamp.outputs[0]

	def math_node(operation, a, b):
		n = tree.nodes.new("ShaderNodeMath")
		n.operation = operation
		for target, value in zip(n.inputs, (a, b)):
			if isinstance(value, (int, float)):
				target.default_value = value
			else:
				tree.links.new(value, target)
		return n.outputs[0]

	if node.type == "MIX_SHADER":
		a, b = (
			coverage(tree, node.inputs[1], notes, seen),
			coverage(tree, node.inputs[2], notes, seen),
		)
		if node.inputs[0].is_linked:
			clamp = tree.nodes.new("ShaderNodeClamp")
			tree.links.new(node.inputs[0].links[0].from_socket, clamp.inputs["Value"])
			f = clamp.outputs[0]
		else:
			f = max(0, min(1, float(node.inputs[0].default_value)))
		return math_node(
			"ADD",
			math_node("MULTIPLY", math_node("SUBTRACT", 1.0, f), a),
			math_node("MULTIPLY", f, b),
		)
	if node.type == "ADD_SHADER":
		notes.append(
			"Add Shader sums transparent extinction; signed coverage a+b-1 is clamped only at the final surface"
		)
		return math_node(
			"SUBTRACT",
			math_node(
				"ADD",
				coverage(tree, node.inputs[0], notes, seen),
				coverage(tree, node.inputs[1], notes, seen),
			),
			1.0,
		)
	if node.type == "GROUP":
		alpha = node.inputs.get("Alpha")
		notes.append(
			"Custom shader group coverage uses exposed Alpha when present; internal layering is approximated"
		)
		if alpha:
			return alpha.links[0].from_socket if alpha.is_linked else float(alpha.default_value)
	return 1.0


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--library", required=True)
	p.add_argument("--object", required=True)
	p.add_argument("--uv", required=True)
	p.add_argument("--manifest")
	a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)
	with bpy.data.libraries.load(a.library, link=False) as (_, target):
		target.objects = [a.object]
	obj = target.objects[0]
	bpy.context.scene.collection.objects.link(obj)
	obj.hide_render = False
	obj.hide_viewport = False
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj
	obj.data.uv_layers.active = obj.data.uv_layers[a.uv]
	obj.data.uv_layers[a.uv].active_render = True
	scene = bpy.context.scene
	scene.render.engine = "CYCLES"
	scene.cycles.device = "CPU"
	scene.cycles.samples = 1
	scene.render.bake.margin = 8
	scene.render.bake.use_clear = True
	if not a.manifest:
		bpy.ops.object.bake(type="EMIT", uv_layer=a.uv)
		for image in {
			s.material.node_tree.nodes.active.image for s in obj.material_slots if s.material
		}:
			image.save()
		return
	spec = json.loads(Path(a.manifest).read_text())
	results = {"channels": {}, "notes": []}
	scene.frame_set(spec.get("frame", 1))
	material = obj.data.materials[0]
	tree = material.node_tree
	shader = tree.nodes[spec["shader"]]
	output = tree.nodes[spec["output"]]
	original = output.inputs["Surface"].links[0].from_socket
	# Evaluated mesh displacement is retained. Shader displacement is explicitly
	# omitted by preparation; do not let it alter only the temporary bake surface.
	for link in list(output.inputs["Displacement"].links):
		tree.links.remove(link)
	alpha = coverage(tree, output.inputs["Surface"], results["notes"])
	coverage_pixels = None
	for channel in spec["channels"]:
		kind = channel["kind"]
		image = bpy.data.images.new(
			"XenonBake_" + kind,
			width=spec["size"],
			height=spec["size"],
			alpha=False,
			float_buffer=True,
		)
		image.colorspace_settings.name = "Non-Color"
		for node in tree.nodes:
			node.select = False
		tex = tree.nodes.new("ShaderNodeTexImage")
		tex.image = image
		tex.select = True
		tree.nodes.active = tex
		tree.links.new(original, output.inputs["Surface"])
		temporary = None
		if kind not in ("normal", "emission"):
			temporary = tree.nodes.new("ShaderNodeEmission")
			if kind == "alpha":
				value = alpha
			else:
				socket = shader.inputs[channel["socket"]]
				value = socket.links[0].from_socket if socket.is_linked else socket.default_value
			if hasattr(value, "node"):
				tree.links.new(value, temporary.inputs["Color"])
			else:
				values = (
					(value, value, value, 1)
					if isinstance(value, (int, float))
					else (*tuple(value)[:3], 1)
				)
				temporary.inputs["Color"].default_value = values
			tree.links.new(temporary.outputs[0], output.inputs["Surface"])
		bpy.ops.object.bake(
			type="NORMAL" if kind == "normal" else "EMIT", uv_layer=a.uv, normal_space="TANGENT"
		)
		pixels = np.empty(len(image.pixels), dtype=np.float32)
		image.pixels.foreach_get(pixels)
		pixels = pixels.reshape((spec["size"], spec["size"], image.channels))
		if not np.all(np.isfinite(pixels)):
			raise RuntimeError(f"{kind}: nonfinite bake values")
		rgb = pixels[:, :, :3]
		entry = {"path": channel["path"], "minimum": float(rgb.min()), "maximum": float(rgb.max())}
		if kind == "emission":
			if entry["minimum"] < -1e-6:
				raise RuntimeError("Negative emitted radiance is unsupported")
			if coverage_pixels is None:
				raise RuntimeError("Coverage must be baked before emission")
			entry["premultiplied_maximum"] = entry["maximum"]
			# Cycles EMIT includes surface closure coverage. The final Principled
			# applies Alpha separately, so store radiance conditional on a hit.
			np.divide(
				rgb, coverage_pixels[:, :, None], out=rgb, where=coverage_pixels[:, :, None] > 0
			)
			rgb[coverage_pixels <= 0] = 0
			if not np.all(np.isfinite(rgb)):
				raise RuntimeError("Nonfinite conditional emission")
			entry["minimum"] = float(rgb.min())
			entry["maximum"] = float(rgb.max())
			image.pixels.foreach_set(pixels.ravel())
			image.filepath_raw = channel["path"]
			image.file_format = "HDR"
			image.save()
		else:
			entry["clamped_components"] = int(np.count_nonzero((rgb < 0) | (rgb > 1)))
			png16(channel["path"], pixels)
			if kind == "alpha":
				coverage_pixels = np.clip(rgb[:, :, 0], 0, 1).copy()
		results["channels"][kind] = entry
		tree.links.new(original, output.inputs["Surface"])
		if temporary:
			tree.nodes.remove(temporary)
		tree.nodes.remove(tex)
		bpy.data.images.remove(image)
	Path(spec["result"]).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
	main()
