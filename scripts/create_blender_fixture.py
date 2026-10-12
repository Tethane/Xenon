#!/usr/bin/env python3
"""Author a deterministic textured material studio in Blender, then export it.

blender --background --factory-startup --python scripts/create_blender_fixture.py -- \
  --output build/fixtures/material_studio --width 512 --height 384 --reference
"""

import math
import argparse
from pathlib import Path
import sys
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export import Exporter, arguments


def principled(name, color, roughness=0.5, metallic=0, transmission=0):
	material = bpy.data.materials.new(name)
	material.use_nodes = True
	node = material.node_tree.nodes.get("Principled BSDF")
	node.inputs["Base Color"].default_value = (*color, 1)
	node.inputs["Roughness"].default_value = roughness
	node.inputs["Metallic"].default_value = metallic
	node.inputs["Transmission Weight"].default_value = transmission
	return material, node


def image(name, size, pixel, colorspace):
	image = bpy.data.images.new(name, width=size, height=size, alpha=True)
	image.colorspace_settings.name = colorspace
	pixels = []
	for y in range(size):
		for x in range(size):
			pixels.extend(pixel(x, y, size))
	image.pixels = pixels
	image.pack()
	return image


def texture(material, image, target, node):
	tex = material.node_tree.nodes.new("ShaderNodeTexImage")
	tex.image = image
	tex.interpolation = "Linear"
	material.node_tree.links.new(tex.outputs["Color"], node.inputs[target])
	return tex


def mesh_emitter(name, location, size, emission, target):
	bpy.ops.mesh.primitive_plane_add(size=2, location=location)
	obj = bpy.context.object
	obj.name = name
	obj.scale = (size[0] / 2, size[1] / 2, 1)
	obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("Z", "Y").to_euler()
	material = bpy.data.materials.new(name)
	material.use_nodes = True
	nodes = material.node_tree.nodes
	nodes.clear()
	output = nodes.new("ShaderNodeOutputMaterial")
	shader = nodes.new("ShaderNodeEmission")
	shader.inputs["Color"].default_value = (*emission, 1)
	shader.inputs["Strength"].default_value = 1
	material.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
	obj.data.materials.append(material)


def create(args, fog=False):
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)
	scene = bpy.context.scene
	scene.unit_settings.system = "METRIC"
	scene.unit_settings.scale_length = 1
	scene.render.resolution_x = args.width or 640
	scene.render.resolution_y = args.height or 480
	scene.render.resolution_percentage = 100
	scene.render.film_transparent = False
	scene.world = bpy.data.worlds.new("Black studio world")
	scene.world.use_nodes = True
	scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
	checker = image(
		"ceramic_checker",
		256,
		lambda x, y, n: (
			(0.68, 0.30, 0.09, 1) if ((x // 32 + y // 32) % 2) else (0.025, 0.17, 0.20, 1)
		),
		"sRGB",
	)
	normal = image(
		"fine_ceramic_normal",
		256,
		lambda x, y, n: (
			0.5 + 0.12 * math.sin(2 * math.pi * x / 16),
			0.5 + 0.12 * math.cos(2 * math.pi * y / 16),
			0.97,
			1,
		),
		"Non-Color",
	)
	rough = image(
		"ceramic_roughness", 256, lambda x, y, n: (0.23 + 0.18 * y / n,) * 3 + (1,), "Non-Color"
	)
	ceramic, node = principled("Textured ceramic", (1, 1, 1), roughness=1)
	texture(ceramic, checker, "Base Color", node)
	texture(ceramic, rough, "Roughness", node)
	normal_node = ceramic.node_tree.nodes.new("ShaderNodeNormalMap")
	normal_node.inputs["Strength"].default_value = 0.22
	texture(ceramic, normal, "Color", normal_node)
	ceramic.node_tree.links.new(normal_node.outputs[0], node.inputs["Normal"])
	floor, _ = principled("Warm matte floor", (0.32, 0.285, 0.23), roughness=0.72)
	teal, _ = principled("Teal platform", (0.013, 0.09, 0.095), roughness=0.34)
	brass, _ = principled("Brushed brass", (0.72, 0.46, 0.12), roughness=0.22, metallic=1)
	glass, _ = principled("Clear glass", (1, 1, 1), roughness=0.035, transmission=1)
	bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.03))
	bpy.context.object.name = "Studio floor"
	bpy.context.object.data.materials.append(floor)
	bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=2.2, depth=0.16, location=(0, 0, 0.05))
	obj = bpy.context.object
	obj.name = "Display plinth"
	obj.data.materials.append(teal)
	bevel = obj.modifiers.new("Softened platform rim", "BEVEL")
	bevel.width = 0.04
	bevel.segments = 3
	obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
	bpy.ops.mesh.primitive_uv_sphere_add(
		segments=64, ring_count=32, radius=0.82, location=(-0.95, 0.0, 0.96)
	)
	obj = bpy.context.object
	obj.name = "Textured ceramic sphere"
	obj.data.materials.append(ceramic)
	for poly in obj.data.polygons:
		poly.use_smooth = True
	bpy.ops.mesh.primitive_torus_add(
		major_segments=64,
		minor_segments=24,
		location=(0.82, 0.3, 0.92),
		major_radius=0.58,
		minor_radius=0.20,
		rotation=(math.pi / 2, 0, 0.35),
	)
	obj = bpy.context.object
	obj.name = "Brass ring"
	obj.data.materials.append(brass)
	for poly in obj.data.polygons:
		poly.use_smooth = True
	bpy.ops.mesh.primitive_uv_sphere_add(
		segments=48, ring_count=24, radius=0.43, location=(0.8, -0.75, 0.57)
	)
	obj = bpy.context.object
	obj.name = "Glass sphere"
	obj.data.materials.append(glass)
	for poly in obj.data.polygons:
		poly.use_smooth = True
	mesh_emitter("Warm key", (-3, -2, 5), (3.5, 3.5), (7.5, 6.2, 4.7), (0, 0, 0))
	mesh_emitter("Cool rim", (2.5, 1.5, 3.4), (2, 3.0), (3.0, 5.3, 7.2), (0, 0, 0.8))
	bpy.ops.object.camera_add(location=(4.7, -7.5, 3.8))
	camera = bpy.context.object
	camera.name = "Studio camera"
	camera.rotation_euler = (
		(Vector((0, 0, 0.85)) - camera.location).to_track_quat("-Z", "Y").to_euler()
	)
	camera.data.lens = 48
	scene.camera = camera
	if fog:
		bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 2))
		fogbox = bpy.context.object
		fogbox.name = "Homogeneous fog boundary"
		fogbox.scale = (3, 3, 2)
		material = bpy.data.materials.new("Homogeneous fog")
		material.use_nodes = True
		nodes = material.node_tree.nodes
		nodes.clear()
		output = nodes.new("ShaderNodeOutputMaterial")
		scatter = nodes.new("ShaderNodeVolumeScatter")
		scatter.inputs["Color"].default_value = (1, 1, 1, 1)
		scatter.inputs["Density"].default_value = 0.065
		scatter.inputs["Anisotropy"].default_value = 0.3
		absorption = nodes.new("ShaderNodeVolumeAbsorption")
		absorption.inputs["Color"].default_value = (0.55, 0.75, 1, 1)
		absorption.inputs["Density"].default_value = 0.02
		add = nodes.new("ShaderNodeAddShader")
		material.node_tree.links.new(scatter.outputs[0], add.inputs[0])
		material.node_tree.links.new(absorption.outputs[0], add.inputs[1])
		material.node_tree.links.new(add.outputs[0], output.inputs["Volume"])
		fogbox.data.materials.append(material)
		camera.data.dof.use_dof = True
		camera.data.dof.aperture_fstop = 0.7
		camera.data.dof.focus_distance = (Vector((0, 0, 0.85)) - camera.location).length
	if getattr(args, "environment_hdr", None):
		environment = scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
		environment.image = bpy.data.images.load(
			str(args.environment_hdr.resolve()), check_existing=True
		)
		background = scene.world.node_tree.nodes.get("Background")
		background.inputs["Strength"].default_value = args.environment_strength
		scene.world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
	root = Path(args.output).resolve()
	root.mkdir(parents=True, exist_ok=True)
	bpy.ops.wm.save_as_mainfile(filepath=str(root / "material_studio.blend"))
	Exporter(args).run()


def fixture_arguments():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--environment-hdr", type=Path)
	parser.add_argument("--environment-strength", type=float, default=0.4)
	options, remaining = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
	args = arguments(remaining)
	args.environment_hdr = options.environment_hdr
	args.environment_strength = options.environment_strength
	return args


if __name__ == "__main__":
	create(fixture_arguments())
