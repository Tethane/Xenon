#!/usr/bin/env python3
"""Explicit measured homogeneous proxies for unsupported Blender volume regions.

This preparation changes an in-memory working scene only. It evaluates density
at a uniform 3D grid, estimates occupancy against the evaluated closed boundary,
and preserves estimated integrated density in an AABB proxy. Overlaps become
disjoint boxes with summed coefficients and a scattering-weighted HG moment.
"""

import itertools
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

PRINCIPLED_REFERENCE = (
	"https://raw.githubusercontent.com/blender/blender/v5.2.2/intern/cycles/kernel/svm/closure.h"
)


def _constant(node, name, color=False, default=None):
	socket = node.inputs.get(name)
	if socket is None:
		return default
	if socket.is_linked:
		raise ValueError(f"{node.name}: linked {name} is not supported by the volume proxy")
	value = socket.default_value
	result = tuple(float(x) for x in value[:3]) if color else float(value)
	if not all(math.isfinite(x) for x in (result if color else (result,))):
		raise ValueError(f"{node.name}: non-finite {name}")
	return result


def _boundary(obj):
	"""Evaluate this volume alone, with render subdivision, avoiding scene scatter."""
	if obj.type != "MESH":
		raise ValueError(f"{obj.name}: only closed mesh volume boundaries can be measured")
	source_scene = bpy.context.window.scene
	temporary = bpy.data.scenes.new("Xenon isolated volume boundary")
	clone = obj.copy()
	temporary.collection.objects.link(clone)
	changes = []
	try:
		for modifier in clone.modifiers:
			if modifier.type == "NODES":
				raise ValueError(
					f"{obj.name}: Geometry Nodes volume boundary requires an explicit render-mesh conversion first"
				)
			modifier.show_viewport = modifier.show_render
			if modifier.type == "SUBSURF":
				changes.append({"modifier": modifier.name, "render_levels": modifier.render_levels})
				modifier.levels = modifier.render_levels
		bpy.context.window.scene = temporary
		temporary.frame_set(source_scene.frame_current)
		graph = bpy.context.evaluated_depsgraph_get()
		evaluated = clone.evaluated_get(graph)
		mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
		try:
			mesh.calc_loop_triangles()
			vertices = [clone.matrix_world @ vertex.co for vertex in mesh.vertices]
			faces = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
			if clone.matrix_world.determinant() < 0:
				faces = [tuple(reversed(face)) for face in faces]
		finally:
			evaluated.to_mesh_clear()
	finally:
		bpy.context.window.scene = source_scene
		bpy.data.scenes.remove(temporary)
		bpy.data.objects.remove(clone, do_unlink=True)
	if not vertices or not faces:
		raise ValueError(f"{obj.name}: empty volume boundary")
	edges = {}
	directions = {}
	for face in faces:
		for a, b in zip(face, face[1:] + face[:1]):
			edge = tuple(sorted((a, b)))
			edges[edge] = edges.get(edge, 0) + 1
			directions[edge] = directions.get(edge, 0) + (1 if a < b else -1)
	if any(count != 2 for count in edges.values()):
		raise ValueError(f"{obj.name}: volume proxy requires a closed manifold boundary")
	if any(direction != 0 for direction in directions.values()):
		raise ValueError(f"{obj.name}: volume proxy requires consistent face winding")
	lower = [min(vertex[a] for vertex in vertices) for a in range(3)]
	upper = [max(vertex[a] for vertex in vertices) for a in range(3)]
	if min(upper[a] - lower[a] for a in range(3)) <= 1e-8:
		raise ValueError(f"{obj.name}: degenerate volume boundary")
	tolerance = max(upper[a] - lower[a] for a in range(3)) * 1e-6
	corners = [
		tuple(
			0 if abs(v[a] - lower[a]) < tolerance else 1 if abs(v[a] - upper[a]) < tolerance else -1
			for a in range(3)
		)
		for v in vertices
	]
	box = (
		len(vertices) == 8
		and len(set(corners)) == 8
		and all(-1 not in corner for corner in corners)
		and all(
			any(len({corners[index][axis] for index in face}) == 1 for axis in range(3))
			for face in faces
		)
	)
	signed_volume = sum(vertices[a].dot(vertices[b].cross(vertices[c])) for a, b, c in faces) / 6
	if signed_volume <= 0:
		raise ValueError(f"{obj.name}: volume boundary must have consistent outward winding")
	return vertices, faces, lower, upper, box, changes


def _density_samples(obj, material, node_name, points, destination):
	"""Use Cycles' shader evaluator, preserving scalar conversion and coordinates."""
	scene_before = bpy.context.window.scene
	scene = bpy.data.scenes.new("Xenon volume density sampling")
	shader = material.copy()
	mesh = bpy.data.meshes.new("Xenon density sample tiles")
	sample_obj = bpy.data.objects.new("Xenon density sample tiles", mesh)
	image = None
	try:
		tree = shader.node_tree
		original_node = tree.nodes.get(node_name)
		density = original_node.inputs["Density"]
		if not density.is_linked:
			return np.full(len(points), max(0, float(density.default_value)))
		reached = set()

		def visit(node):
			if node.as_pointer() in reached:
				return
			reached.add(node.as_pointer())
			for socket in node.inputs:
				for link in socket.links:
					visit(link.from_node)

		visit(density.links[0].from_node)
		source_output = density.links[0].from_socket
		columns = 2 ** math.ceil(math.log2(math.sqrt(len(points))))
		rows = math.ceil(len(points) / columns)
		verts, faces, uvs = [], [], []
		for i in range(len(points)):
			x, y = i % columns, i // columns
			verts.extend(((x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)))
			faces.append(tuple(range(4 * i, 4 * i + 4)))
			uvs.extend(
				(
					(x / columns, y / rows),
					((x + 1) / columns, y / rows),
					((x + 1) / columns, (y + 1) / rows),
					(x / columns, (y + 1) / rows),
				)
			)
		mesh.from_pydata(verts, [], faces)
		uv = mesh.uv_layers.new(name="Density sample tiles")
		uv.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
		inverse = obj.matrix_world.inverted()
		local_points = [inverse @ Vector(point) for point in points]
		# Generated coordinates use the original object's texture space,
		# before modifiers, rather than the sample-tile mesh's bounding box.
		tex_location, tex_size = obj.data.texspace_location, obj.data.texspace_size
		generated = [
			[
				(point[a] - tex_location[a]) / (2 * tex_size[a]) + 0.5
				if abs(tex_size[a]) > 1e-12
				else 0.5
				for a in range(3)
			]
			for point in local_points
		]

		def attribute(name, values):
			attr = mesh.attributes.new(name, "FLOAT_VECTOR", "POINT")
			attr.data.foreach_set(
				"vector", np.repeat(np.asarray(values, dtype=np.float32), 4, axis=0).ravel()
			)
			node = tree.nodes.new("ShaderNodeAttribute")
			node.attribute_name = name
			return node.outputs["Vector"]

		generated_socket = attribute("xenon_generated", generated)
		object_socket = attribute("xenon_object", local_points)
		position_socket = attribute("xenon_position", points)
		for node in list(tree.nodes):
			if node.as_pointer() not in reached:
				continue
			if node.type in {
				"GROUP",
				"ATTRIBUTE",
				"OBJECT_INFO",
				"LIGHT_PATH",
				"TEX_IMAGE",
				"TEX_ENVIRONMENT",
				"UVMAP",
			}:
				raise ValueError(
					f"{obj.name}: density node {node.bl_idname} requires unsupported sampling context"
				)
			if node.type == "TEX_COORD":
				for output in node.outputs:
					if not output.is_linked:
						continue
					replacements = {"Generated": generated_socket, "Object": object_socket}
					if output.name not in replacements or (output.name == "Object" and node.object):
						raise ValueError(
							f"{obj.name}: density coordinate {output.name} is unsupported"
						)
					for link in list(output.links):
						tree.links.new(replacements[output.name], link.to_socket)
			elif node.type == "NEW_GEOMETRY":
				for output in node.outputs:
					if output.is_linked:
						if output.name != "Position":
							raise ValueError(
								f"{obj.name}: density geometry output {output.name} is unsupported"
							)
						for link in list(output.links):
							tree.links.new(position_socket, link.to_socket)
			elif (
				node.type.startswith("TEX_")
				and node.inputs.get("Vector")
				and not node.inputs["Vector"].is_linked
			):
				tree.links.new(generated_socket, node.inputs["Vector"])
		output = next(
			node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output
		)
		for socket in output.inputs:
			for link in list(socket.links):
				tree.links.remove(link)
		clamp = tree.nodes.new("ShaderNodeMath")
		clamp.operation = "MAXIMUM"
		clamp.inputs[1].default_value = 0
		tree.links.new(source_output, clamp.inputs[0])
		emission = tree.nodes.new("ShaderNodeEmission")
		tree.links.new(clamp.outputs[0], emission.inputs["Color"])
		tree.links.new(emission.outputs[0], output.inputs["Surface"])
		image = bpy.data.images.new(
			"Measured volume density", width=columns, height=rows, float_buffer=True
		)
		image.colorspace_settings.name = "Non-Color"
		target = tree.nodes.new("ShaderNodeTexImage")
		target.image = image
		tree.nodes.active = target
		mesh.materials.append(shader)
		scene.collection.objects.link(sample_obj)
		bpy.context.window.scene = scene
		scene.frame_set(scene_before.frame_current)
		bpy.context.view_layer.objects.active = sample_obj
		sample_obj.select_set(True)
		scene.render.engine = "CYCLES"
		scene.cycles.device = "CPU"
		scene.cycles.samples = 1
		scene.render.threads_mode = "FIXED"
		scene.render.threads = 2
		scene.render.bake.margin = 0
		scene.render.bake.use_clear = True
		scene.render.bake.use_selected_to_active = False
		bpy.ops.object.bake(type="EMIT")
		pixels = np.asarray(image.pixels[:], dtype=np.float64).reshape(-1, 4)[: len(points)]
		if not np.isfinite(pixels).all() or not np.all(pixels[:, 3] > 0.99):
			raise ValueError(f"{obj.name}: incomplete or non-finite density bake")
		image.filepath_raw = str(destination)
		image.file_format = "OPEN_EXR"
		image.save()
		return np.maximum(pixels[:, 0], 0)
	finally:
		bpy.context.window.scene = scene_before
		bpy.data.scenes.remove(scene)
		bpy.data.objects.remove(sample_obj, do_unlink=True)
		bpy.data.meshes.remove(mesh)
		bpy.data.materials.remove(shader)
		if image:
			bpy.data.images.remove(image)


def partition(regions):
	"""Create disjoint cells; sum extinction and preserve the scalar HG first moment."""
	planes = [
		sorted({region[key][a] for region in regions for key in ("lower", "upper")})
		for a in range(3)
	]
	if math.prod(len(axis) - 1 for axis in planes) > 100000:
		raise ValueError("Too many overlapping volume boundaries for explicit proxy partition")
	cells = []
	for indices in itertools.product(*(range(len(axis) - 1) for axis in planes)):
		lower = [planes[a][indices[a]] for a in range(3)]
		upper = [planes[a][indices[a] + 1] for a in range(3)]
		center = [(a + b) / 2 for a, b in zip(lower, upper)]
		active = [
			i
			for i, region in enumerate(regions)
			if all(region["lower"][a] < center[a] < region["upper"][a] for a in range(3))
		]
		if not active:
			continue
		sigma_a = [sum(regions[i]["sigma_a"][a] for i in active) for a in range(3)]
		sigma_s = [sum(regions[i]["sigma_s"][a] for i in active) for a in range(3)]
		weight = sum(sum(regions[i]["sigma_s"]) for i in active)
		g = (
			sum(sum(regions[i]["sigma_s"]) * regions[i]["g"] for i in active) / weight
			if weight
			else 0
		)
		cells.append(
			{
				"lower": lower,
				"upper": upper,
				"sigma_a": sigma_a,
				"sigma_s": sigma_s,
				"g": g,
				"sources": active,
			}
		)
	# Adjacent regions with identical contributing media can be represented by
	# one box without changing any coefficients or introducing overlap.
	changed = True
	while changed:
		changed = False
		for i, first in enumerate(cells):
			for j in range(i + 1, len(cells)):
				second = cells[j]
				if first["sources"] != second["sources"]:
					continue
				for axis in range(3):
					others = [a for a in range(3) if a != axis]
					same = all(
						first["lower"][a] == second["lower"][a]
						and first["upper"][a] == second["upper"][a]
						for a in others
					)
					touches = (
						first["upper"][axis] == second["lower"][axis]
						or second["upper"][axis] == first["lower"][axis]
					)
					if same and touches:
						first["lower"][axis] = min(first["lower"][axis], second["lower"][axis])
						first["upper"][axis] = max(first["upper"][axis], second["upper"][axis])
						cells.pop(j)
						changed = True
						break
				if changed:
					break
			if changed:
				break
	return cells


def _box(scene, cell, index):
	material = bpy.data.materials.new(f"Xenon volume proxy {index}")
	material.use_nodes = True
	tree = material.node_tree
	tree.nodes.clear()
	output = tree.nodes.new("ShaderNodeOutputMaterial")
	closures = []
	for field, kind in (
		("sigma_s", "ShaderNodeVolumeScatter"),
		("sigma_a", "ShaderNodeVolumeAbsorption"),
	):
		coefficients = cell[field]
		density = max(coefficients)
		if density <= 0:
			continue
		node = tree.nodes.new(kind)
		color = [x / density for x in coefficients]
		if field == "sigma_a":
			color = [1 - x for x in color]
		else:
			node.inputs["Anisotropy"].default_value = cell["g"]
		node.inputs["Density"].default_value = density
		node.inputs["Color"].default_value = (*color, 1)
		closures.append(node.outputs[0])
	if not closures:
		bpy.data.materials.remove(material)
		return None
	if len(closures) == 2:
		add = tree.nodes.new("ShaderNodeAddShader")
		for i, closure in enumerate(closures):
			tree.links.new(closure, add.inputs[i])
		surface = add.outputs[0]
	else:
		surface = closures[0]
	tree.links.new(surface, output.inputs["Volume"])
	vertices = [
		tuple(cell["upper"][a] if bit[a] else cell["lower"][a] for a in range(3))
		for bit in itertools.product((0, 1), repeat=3)
	]
	faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
	mesh = bpy.data.meshes.new(f"Xenon volume proxy {index}")
	mesh.from_pydata(vertices, [], faces)
	mesh.materials.append(material)
	obj = bpy.data.objects.new(mesh.name, mesh)
	scene.collection.objects.link(obj)
	return obj


def prepare_volumes(source, output_dir, grid_size=16):
	if not 2 <= grid_size <= 32:
		raise ValueError("Volume density sampling grid must be between 2 and 32 per axis")
	if abs(source.unit_settings.scale_length - 1) > 1e-8:
		raise ValueError("Volume proxies require one Blender unit per meter")
	destination = Path(output_dir).resolve() / "volumes"
	destination.mkdir(parents=True, exist_ok=True)
	report = {
		"status": "in_progress",
		"source": bpy.data.filepath,
		"frame": source.frame_current,
		"blender_version": bpy.app.version_string,
		"grid_size": grid_size,
		"principled_coefficients_reference": PRINCIPLED_REFERENCE,
		"regions": [],
		"partition": [],
		"notes": [],
	}
	objects = []
	needs_conversion = False
	visible_objects = set()
	view_layer = bpy.context.view_layer if bpy.context.scene == source else source.view_layers[0]

	def visible_collections(layer):
		if layer.exclude or layer.collection.hide_render:
			return
		visible_objects.update(obj.as_pointer() for obj in layer.collection.objects)
		for child in layer.children:
			visible_collections(child)

	visible_collections(view_layer.layer_collection)
	for obj in list(source.objects):
		if obj.hide_render or obj.as_pointer() not in visible_objects:
			continue
		found = []
		for slot in obj.material_slots:
			material = slot.material
			if material and material.node_tree:
				output = next(
					(
						n
						for n in material.node_tree.nodes
						if n.type == "OUTPUT_MATERIAL" and n.is_active_output
					),
					None,
				)
				if output and output.inputs["Volume"].is_linked:
					found.append((material, output))
		if not found:
			continue
		if len(found) != 1 or len(obj.material_slots) != 1:
			raise ValueError(
				f"{obj.name}: proxy requires one volume material over its entire boundary"
			)
		material, output = found[0]
		if output.inputs["Surface"].is_linked or output.inputs["Displacement"].is_linked:
			raise ValueError(
				f"{obj.name}: volume proxy cannot remove an accompanying surface or displacement shader"
			)
		node = output.inputs["Volume"].links[0].from_node
		if node.type not in {
			"PRINCIPLED_VOLUME",
			"VOLUME_PRINCIPLED",
			"VOLUME_SCATTER",
			"VOLUME_ABSORPTION",
		}:
			raise ValueError(f"{obj.name}: unsupported volume closure {node.bl_idname}")
		principled = node.bl_idname == "ShaderNodeVolumePrincipled"
		if (
			node.type == "VOLUME_SCATTER"
			and getattr(node, "phase", "HENYEY_GREENSTEIN") != "HENYEY_GREENSTEIN"
		):
			raise ValueError(f"{obj.name}: only HG volume phase functions can be proxied")
		color = _constant(node, "Color", color=True)
		if any(c < 0 or c > 1 for c in color):
			raise ValueError(f"{obj.name}: volume color outside [0,1]")
		g = _constant(node, "Anisotropy", default=0)
		if not -1 < g < 1:
			raise ValueError(f"{obj.name}: volume anisotropy outside (-1,1)")
		absorption = (0, 0, 0)
		if principled:
			if (
				_constant(node, "Emission Strength") != 0
				or _constant(node, "Blackbody Intensity") != 0
			):
				raise ValueError(f"{obj.name}: emissive volume proxies are unsupported")
			absorption = _constant(node, "Absorption Color", color=True)
			if any(c < 0 or c > 1 for c in absorption):
				raise ValueError(f"{obj.name}: absorption color outside [0,1]")
			for name in ("Density Attribute", "Color Attribute"):
				attr = node.inputs.get(name)
				if attr and attr.default_value and obj.data.attributes.get(attr.default_value):
					raise ValueError(
						f"{obj.name}: named volume attributes must be baked explicitly first"
					)
		vertices, faces, lower, upper, box, modifier_changes = _boundary(obj)
		density_socket = node.inputs["Density"]
		procedural = density_socket.is_linked
		approximate = procedural or not box
		needs_conversion |= principled or approximate
		region = {
			"object": obj.name,
			"material": material.name,
			"closure": node.bl_idname,
			"lower": lower,
			"upper": upper,
			"g": g,
			"color": color,
			"absorption_color": absorption,
			"boundary_vertices": len(vertices),
			"boundary_triangles": len(faces),
			"render_modifier_evaluation": modifier_changes,
			"source_was_axis_aligned_box": box,
			"procedural_density": procedural,
			"approximation": approximate,
		}
		if approximate:
			points = [
				[
					lower[a] + (indices[a] + 0.5) * (upper[a] - lower[a]) / grid_size
					for a in range(3)
				]
				for indices in itertools.product(range(grid_size), repeat=3)
			]
			bvh = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
			mask = []
			for point in points:
				position, normal, _, _ = bvh.find_nearest(Vector(point))
				if position is None:
					raise ValueError(f"{obj.name}: occupancy evaluation failed")
				mask.append((Vector(point) - position).dot(normal) <= 0)
			mask = np.asarray(mask, dtype=bool)
			if not mask.any():
				raise ValueError(
					f"{obj.name}: sampling grid missed its interior; increase grid_size"
				)
			if procedural:
				samples = _density_samples(
					obj,
					material,
					node.name,
					points,
					destination / f"density_{len(objects):02d}.exr",
				)
			else:
				samples = np.full(len(points), max(0, _constant(node, "Density")))
			mean = float(samples[mask].mean())
			occupancy = float(mask.mean())
			density = mean * occupancy
			region.update(
				grid_size=grid_size,
				sample_count=len(points),
				inside_count=int(mask.sum()),
				mean_inside_density=mean,
				occupancy=occupancy,
				density_in_proxy=density,
				minimum_inside_density=float(samples[mask].min()),
				maximum_inside_density=float(samples[mask].max()),
				measured_integrated_density=density
				* math.prod(upper[a] - lower[a] for a in range(3)),
			)
			(destination / f"samples_{len(objects):02d}.json").write_text(
				json.dumps(
					{"points": points, "inside": mask.tolist(), "density": samples.tolist()},
					indent=2,
				)
				+ "\n"
			)
		else:
			density = max(0, _constant(node, "Density"))
			region.update(mean_inside_density=density, occupancy=1.0, density_in_proxy=density)
		if principled:
			region["sigma_s"] = [density * c for c in color]
			region["sigma_a"] = [
				density * (1 - c) * (1 - math.sqrt(a)) for c, a in zip(color, absorption)
			]
		elif node.type == "VOLUME_SCATTER":
			region["sigma_s"] = [density * c for c in color]
			region["sigma_a"] = [0, 0, 0]
		else:
			region["sigma_s"] = [0, 0, 0]
			region["sigma_a"] = [density * (1 - c) for c in color]
		report["regions"].append(region)
		objects.append(obj)
	if needs_conversion:
		report["partition"] = partition(report["regions"])
		for i, cell in enumerate(report["partition"]):
			proxy = _box(source, cell, i)
			cell["object"] = proxy.name if proxy else None
		for obj in objects:
			obj.hide_render = True
		report["notes"].append(
			"Spatial density and non-box boundaries become homogeneous AABB proxies; finite-grid estimated integrated density is preserved, not ray-by-ray optical depth"
		)
		if any(len(cell["sources"]) > 1 for cell in report["partition"]):
			report["notes"].append(
				"Overlapping coefficients are summed in disjoint cells; multiple HG lobes become one scattering-RGB-sum-weighted first moment, an angular and spectral approximation"
			)
	report["status"] = "converted" if needs_conversion else "already_supported"
	(destination / "volume-report.json").write_text(json.dumps(report, indent=2) + "\n")
	return report
