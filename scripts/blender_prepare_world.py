#!/usr/bin/env python3
"""Bake a Blender world into a scene-linear equirectangular environment.

Run inside Blender, or import prepare_world from the scene preparation script.
The supplied scene is a derived working copy: its world is replaced, while the
original world's shader data remains intact. Ray-dependent worlds use the
documented diffuse-reflection branch because Xenon has one environment for all
ray types. World volumes are rejected instead of being discarded.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy


DIFFUSE_RAY = {
	"Is Camera Ray": 0.0,
	"Is Shadow Ray": 0.0,
	"Is Diffuse Ray": 1.0,
	"Is Glossy Ray": 0.0,
	"Is Singular Ray": 0.0,
	"Is Reflection Ray": 1.0,
	"Is Transmission Ray": 0.0,
}


def _freeze_light_path(tree, report, copied_groups):
	for node in list(tree.nodes):
		if node.type == "GROUP" and node.node_tree:
			original = node.node_tree
			key = original.as_pointer()
			if key not in copied_groups:
				copied_groups[key] = original.copy()
				_freeze_light_path(copied_groups[key], report, copied_groups)
			node.node_tree = copied_groups[key]
		if node.type != "LIGHT_PATH":
			continue
		for output in node.outputs:
			if not output.is_linked:
				continue
			if output.name not in DIFFUSE_RAY:
				raise ValueError(
					f"World uses contextual Light Path output {output.name}; a direction-only environment cannot preserve it"
				)
			replacement = tree.nodes.new("ShaderNodeValue")
			replacement.label = "Xenon diffuse environment branch: " + output.name
			replacement.outputs[0].default_value = DIFFUSE_RAY[output.name]
			for link in list(output.links):
				tree.links.new(replacement.outputs[0], link.to_socket)
			report["light_path_changes"].append(
				{
					"node_tree": tree.name,
					"node": node.name,
					"output": output.name,
					"value": DIFFUSE_RAY[output.name],
				}
			)


def prepare_world(scene, output_dir, width=2048, height=1024, samples=32, device="CPU"):
	"""Replace scene.world with its baked diffuse-lighting environment; return provenance."""
	if width < 4 or height < 2 or width != 2 * height or samples < 1:
		raise ValueError("World bake requires positive samples and 2:1 dimensions of at least 4x2")
	if device != "CPU":
		raise ValueError(
			"World bake currently uses CPU for reproducibility and to avoid GPU contention"
		)
	original = scene.world
	report = {
		"source_world": original.name if original else None,
		"method": "Cycles empty-scene equirectangular panorama",
		"width": width,
		"height": height,
		"samples": samples,
		"device": device,
		"blender_version": bpy.app.version_string,
		"color_space": "scene-linear Rec.709",
		"light_path_changes": [],
		"notes": [],
	}
	if original is None:
		report["status"] = "no_world"
		return report
	if original.use_nodes and original.node_tree:
		output = next(
			(
				node
				for node in original.node_tree.nodes
				if node.type == "OUTPUT_WORLD" and node.is_active_output
			),
			None,
		)
		if output and output.inputs["Volume"].is_linked:
			raise ValueError("World volumes cannot be converted into an environment image")
	destination = Path(output_dir).resolve() / "world"
	destination.mkdir(parents=True, exist_ok=True)
	target = destination / "environment.hdr"
	bake_scene = bpy.data.scenes.new("Xenon world panorama bake")
	bake_world = original.copy()
	camera_data = bpy.data.cameras.new("Xenon environment panorama")
	camera = bpy.data.objects.new("Xenon environment panorama", camera_data)
	copied_groups = {}
	try:
		if not bake_world.use_nodes:
			color = tuple(bake_world.color)
			bake_world.use_nodes = True
			background = next(
				node for node in bake_world.node_tree.nodes if node.type == "BACKGROUND"
			)
			background.inputs["Color"].default_value = (*color, 1)
		_freeze_light_path(bake_world.node_tree, report, copied_groups)
		bake_scene.world = bake_world
		bake_scene.frame_set(scene.frame_current)
		bake_scene.render.engine = "CYCLES"
		bake_scene.cycles.device = "CPU"
		bake_scene.cycles.samples = samples
		bake_scene.cycles.use_denoising = False
		bake_scene.cycles.use_adaptive_sampling = False
		bake_scene.cycles.seed = 1
		bake_scene.cycles.pixel_filter_type = "BOX"
		bake_scene.render.resolution_x = width
		bake_scene.render.resolution_y = height
		bake_scene.render.resolution_percentage = 100
		bake_scene.render.threads_mode = "FIXED"
		bake_scene.render.threads = 2
		bake_scene.render.film_transparent = False
		bake_scene.render.use_compositing = False
		bake_scene.render.use_sequencer = False
		bake_scene.render.dither_intensity = 0
		bake_scene.render.image_settings.file_format = "HDR"
		bake_scene.render.image_settings.color_mode = "RGB"
		bake_scene.render.filepath = str(target)
		bake_scene.view_settings.view_transform = "Raw"
		bake_scene.view_settings.look = "None"
		bake_scene.view_settings.exposure = 0
		bake_scene.view_settings.gamma = 1
		camera_data.type = "PANO"
		camera_data.panorama_type = "EQUIRECTANGULAR"
		camera_data.longitude_min = -math.pi
		camera_data.longitude_max = math.pi
		camera_data.latitude_min = -math.pi / 2
		camera_data.latitude_max = math.pi / 2
		# Center points toward +X, increasing horizontal coordinate points -Y,
		# and the image top is +Z: Blender's untransformed Environment mapping.
		camera.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
		bake_scene.collection.objects.link(camera)
		bake_scene.camera = camera
		bpy.ops.render.render(scene=bake_scene.name, write_still=True)
		if not target.is_file():
			raise RuntimeError("Cycles did not write the world environment image")
		image = bpy.data.images.load(str(target), check_existing=False)
		image.colorspace_settings.name = "Non-Color"
		replacement = bpy.data.worlds.new(original.name + " [Xenon environment bake]")
		replacement.use_nodes = True
		tree = replacement.node_tree
		tree.nodes.clear()
		environment = tree.nodes.new("ShaderNodeTexEnvironment")
		environment.image = image
		environment.projection = "EQUIRECTANGULAR"
		environment.interpolation = "Linear"
		background = tree.nodes.new("ShaderNodeBackground")
		background.inputs["Strength"].default_value = 1
		output = tree.nodes.new("ShaderNodeOutputWorld")
		tree.links.new(environment.outputs["Color"], background.inputs["Color"])
		tree.links.new(background.outputs[0], output.inputs["Surface"])
		scene.world = replacement
		report.update(
			status="baked", path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()
		)
		report["notes"].append(
			"Finite-resolution RGBE environment approximates the original directional radiance; small sun discs and high-frequency detail are filtered"
		)
		if report["light_path_changes"]:
			report["notes"].append(
				"All rays now see the diffuse-reflection lighting branch; the source camera/glossy background may differ"
			)
		(destination / "world-report.json").write_text(json.dumps(report, indent=2) + "\n")
		return report
	finally:
		bpy.data.scenes.remove(bake_scene)
		bpy.data.objects.remove(camera, do_unlink=True)
		bpy.data.cameras.remove(camera_data)
		bpy.data.worlds.remove(bake_world)
		for group in copied_groups.values():
			bpy.data.node_groups.remove(group)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--width", type=int, default=2048)
	parser.add_argument("--samples", type=int, default=32)
	parser.add_argument("--save-blend", type=Path)
	args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
	if args.save_blend and args.save_blend.resolve() == Path(bpy.data.filepath).resolve():
		parser.error("--save-blend must preserve the opened original")
	report = prepare_world(
		bpy.context.scene, args.output, args.width, args.width // 2, args.samples
	)
	if args.save_blend:
		args.save_blend.parent.mkdir(parents=True, exist_ok=True)
		bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()
