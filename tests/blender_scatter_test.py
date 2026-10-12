#!/usr/bin/env python3
"""Focused export-integrity checks in Blender; no Cycles or GPU render required.

blender --background --factory-startup --threads 2 --python-exit-code 1 \
  --python tests/blender_scatter_test.py
"""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import Exporter


render_objects = []


class CaptureRender(bpy.types.RenderEngine):
	bl_idname = "XENON_SCATTER_CAPTURE"
	bl_label = "Xenon scatter evaluation check"
	bl_use_preview = False

	def render(self, depsgraph):
		# Independent oracle: Blender evaluates a RENDER dependency graph.
		# No shading or sample integration takes place.
		assert depsgraph.mode == "RENDER"
		render_objects[:] = [
			(instance.object.name, instance.is_instance)
			for instance in depsgraph.object_instances
			if instance.show_self
			and instance.object.type == "MESH"
			and len(instance.object.data.polygons) > 0
		]
		result = self.begin_result(0, 0, 1, 1)
		self.end_result(result)


bpy.utils.register_class(CaptureRender)


def setup():
	bpy.ops.wm.read_factory_settings(use_empty=True)
	scene = bpy.context.scene
	scene.world = None
	scene.render.engine = "XENON_SCATTER_CAPTURE"
	scene.render.resolution_x = 4
	scene.render.resolution_y = 4
	scene.render.resolution_percentage = 100
	bpy.ops.object.camera_add(location=(0, -6, 4))
	camera = bpy.context.object
	camera.rotation_euler = (
		(Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
	)
	scene.camera = camera
	material = bpy.data.materials.new("Plain")
	material.use_nodes = True
	mesh = bpy.data.meshes.new("Blade")
	mesh.from_pydata([(0, 0, 0), (0.1, 0, 0), (0, 0, 1)], [], [(0, 1, 2)])
	mesh.materials.append(material)
	blade = bpy.data.objects.new("Blade", mesh)
	scene.collection.objects.link(blade)
	blade.location = (10, 0, 0)
	bpy.ops.mesh.primitive_plane_add(size=2)
	emitter = bpy.context.object
	emitter.name = "Emitter"
	emitter.data.materials.append(material)
	return scene, emitter, blade


def particle(emitter, blade):
	bpy.context.view_layer.objects.active = emitter
	bpy.ops.object.particle_system_add()
	settings = emitter.particle_systems[-1].settings
	settings.type = "HAIR"
	settings.count = 20
	settings.hair_length = 1
	settings.render_type = "OBJECT"
	settings.instance_object = blade
	settings.display_percentage = 100
	return settings


def scatter(emitter, blade):
	group = bpy.data.node_groups.new("Scatter", "GeometryNodeTree")
	group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
	group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
	modifier = emitter.modifiers.new("Scatter", "NODES")
	modifier.node_group = group
	output = group.nodes.new("NodeGroupOutput")
	points = group.nodes.new("GeometryNodeMeshLine")
	points.inputs["Count"].default_value = 3
	info = group.nodes.new("GeometryNodeObjectInfo")
	info.inputs["Object"].default_value = blade
	info.inputs["As Instance"].default_value = True
	instances = group.nodes.new("GeometryNodeInstanceOnPoints")
	group.links.new(points.outputs["Mesh"], instances.inputs["Points"])
	group.links.new(info.outputs["Geometry"], instances.inputs["Instance"])
	group.links.new(instances.outputs["Instances"], output.inputs["Geometry"])
	return group, points


def export(path, native=True):
	bpy.context.view_layer.update()
	args = SimpleNamespace(
		output=str(path), width=4, height=4, samples=1, instances=native, reference=False
	)
	Exporter(args).run()
	return json.loads((path / "export-report.json").read_text())


def reject(path, message):
	try:
		export(path)
	except RuntimeError:
		pass
	else:
		raise AssertionError("Unsupported scene was accepted: " + message)
	report = json.loads((path / "export-report.json").read_text())
	assert report["status"] == "unsupported"
	assert any(message in error["message"] for error in report["errors"]), report["errors"]
	assert not (path / "scene.xenon").exists(), "Rejected export left a loadable scene"


def matches_render(path):
	bpy.context.scene.frame_set(2)
	bpy.context.scene.frame_set(1)
	bpy.ops.render.render()
	expected = sorted(render_objects)
	for native in (False, True):
		report = export(path / str(native), native)
		actual = sorted((obj["name"], obj["instance"]) for obj in report["objects"])
		assert actual == expected, (actual, expected)
	return expected


def move_to(obj, collection):
	for linked in list(obj.users_collection):
		linked.objects.unlink(obj)
	collection.objects.link(obj)


def layer_for(collection):
	def find(layer):
		if layer.collection == collection:
			return layer
		for child in layer.children:
			result = find(child)
			if result:
				return result

	return find(bpy.context.view_layer.layer_collection)


with tempfile.TemporaryDirectory(prefix="xenon_scatter_test_") as tmp:
	root = Path(tmp)
	scene, emitter, blade = setup()
	if hasattr(bpy.types, "ParticleSettings"):
		settings = particle(emitter, blade)
		blade.hide_render = True
		expected = matches_render(root / "hidden-particle-prototype")
		assert expected.count(("Blade", True)) == 20 and ("Blade", False) not in expected
		# The emitter surface can be hidden independently of its particles.
		emitter.show_instancer_for_render = False
		emitter.show_instancer_for_viewport = False
		expected = matches_render(root / "hidden-emitter-mesh")
		assert ("Emitter", False) not in expected and expected.count(("Blade", True)) == 20
		settings.display_percentage = 25
		reject(root / "reduced-particles", "viewport density")
		settings.display_percentage = 100
		settings.child_type = "SIMPLE"
		settings.child_percent = 2
		settings.rendered_child_count = 7
		reject(root / "different-children", "child counts differ")
		settings.child_type = "NONE"
		emitter.show_instancer_for_render = True
		reject(
			root / "different-emitter-visibility", "Instancer viewport/render visibility differs"
		)
		emitter.show_instancer_for_render = False
		# Raw paths were previously omitted even with a hidden emitter surface.
		path = root / "raw-hair"
		export(path)
		settings.render_type = "PATH"
		reject(path, "raw particle hair/paths are unsupported")
		# Disabled particle modifiers contribute neither instances nor raw hair.
		modifier = next(m for m in emitter.modifiers if m.type == "PARTICLE_SYSTEM")
		modifier.show_render = False
		modifier.show_viewport = False
		export(root / "disabled-hair")
	else:
		print("Legacy particle system API absent; particle cases require Blender 4.5")

	scene, emitter, blade = setup()
	blade.hide_render = True
	group, points = scatter(emitter, blade)
	expected = matches_render(root / "hidden-gn-prototype")
	assert expected.count(("Blade", True)) == 3 and ("Blade", False) not in expected
	# Also inspect a nested node group, rather than only top-level graph nodes.
	nested = bpy.data.node_groups.new("RenderMode", "GeometryNodeTree")
	nested.interface.new_socket(name="Viewport", in_out="OUTPUT", socket_type="NodeSocketBool")
	nested_output = nested.nodes.new("NodeGroupOutput")
	viewport = nested.nodes.new("GeometryNodeIsViewport")
	nested.links.new(viewport.outputs[0], nested_output.inputs["Viewport"])
	call = group.nodes.new("GeometryNodeGroup")
	call.node_tree = nested
	switch = group.nodes.new("GeometryNodeSwitch")
	switch.input_type = "INT"
	switch.inputs["False"].default_value = 3
	switch.inputs["True"].default_value = 9
	group.links.new(call.outputs[0], switch.inputs["Switch"])
	group.links.new(switch.outputs[0], points.inputs["Count"])
	reject(root / "render-mode-density", "uses Is Viewport")

	scene, emitter, blade = setup()
	blade.hide_render = True
	collection = bpy.data.collections.new("Instances")
	collection.objects.link(blade)
	instancer = bpy.data.objects.new("Instancer", None)
	scene.collection.objects.link(instancer)
	instancer.instance_type = "COLLECTION"
	instancer.instance_collection = collection
	expected = matches_render(root / "hidden-collection-member")
	assert not any(name == "Blade" for name, _ in expected)

	scene, emitter, blade = setup()
	blade.hide_render = True
	scatter(emitter, blade)
	emitter.hide_render = True
	assert matches_render(root / "hidden-instancer") == []

	# Collection render flags differ from object flags in the viewport graph.
	scene, emitter, blade = setup()
	hidden = bpy.data.collections.new("Render-hidden")
	scene.collection.children.link(hidden)
	child = bpy.data.collections.new("Child")
	hidden.children.link(child)
	move_to(emitter, child)
	hidden.hide_render = True
	expected = matches_render(root / "hidden-ancestor-collection")
	assert expected == [("Blade", False)]
	# An unrelated visible link keeps the same object visible.
	scene.collection.objects.link(emitter)
	expected = matches_render(root / "multiple-collection-links")
	assert ("Emitter", False) in expected
	scene.collection.objects.unlink(emitter)
	hidden.hide_render = False
	layer_for(hidden).exclude = True
	assert matches_render(root / "excluded-layer-collection") == [("Blade", False)]

	# Hidden collections can store prototypes used by visible generators.
	for mechanism in ("geometry-nodes", "particles"):
		if mechanism == "particles" and not hasattr(bpy.types, "ParticleSettings"):
			continue
		scene, emitter, blade = setup()
		prototypes = bpy.data.collections.new("Hidden prototypes")
		scene.collection.children.link(prototypes)
		move_to(blade, prototypes)
		prototypes.hide_render = True
		if mechanism == "geometry-nodes":
			scatter(emitter, blade)
			count = 3
		else:
			particle(emitter, blade)
			count = 20
		expected = matches_render(root / ("collection-prototype-" + mechanism))
		assert expected.count(("Blade", True)) == count and ("Blade", False) not in expected
		hidden = bpy.data.collections.new("Hidden generator")
		scene.collection.children.link(hidden)
		move_to(emitter, hidden)
		hidden.hide_render = True
		assert matches_render(root / ("collection-generator-" + mechanism)) == []

	# Collection instances respect their own nested collection render flags.
	scene, emitter, blade = setup()
	prototype = bpy.data.collections.new("Instance root")
	nested = bpy.data.collections.new("Hidden instance child")
	prototype.children.link(nested)
	move_to(blade, nested)
	nested.hide_render = True
	instancer = bpy.data.objects.new("Instancer", None)
	scene.collection.objects.link(instancer)
	instancer.instance_type = "COLLECTION"
	instancer.instance_collection = prototype
	assert matches_render(root / "hidden-instance-collection") == [("Emitter", False)]
	nested.hide_render = False
	assert ("Blade", True) in matches_render(root / "visible-instance-collection")
	# Excluding a scene link to a prototype does not exclude its instance.
	scene.collection.children.link(prototype)
	layer_for(prototype).exclude = True
	expected = matches_render(root / "excluded-prototype-visible-instance")
	assert ("Blade", True) in expected and ("Blade", False) not in expected

	# A nested collection instancer may point outside the scene collection tree.
	inner = bpy.data.collections.new("Inner instance root")
	inner.children.link(nested)
	prototype.children.unlink(nested)
	middle = bpy.data.objects.new("Middle instance", None)
	prototype.objects.link(middle)
	middle.instance_type = "COLLECTION"
	middle.instance_collection = inner
	assert ("Blade", True) in matches_render(root / "nested-collection-instances")

	# The same collection filter applies to lighting, not only geometry.
	scene, emitter, blade = setup()
	hidden = bpy.data.collections.new("Hidden lights")
	scene.collection.children.link(hidden)
	light_data = bpy.data.lights.new("Sun", "SUN")
	light_data.angle = 0
	light = bpy.data.objects.new("Sun", light_data)
	hidden.objects.link(light)
	hidden.hide_render = True
	export(root / "hidden-collection-light")
	assert "sunlight " not in (root / "hidden-collection-light" / "scene.xenon").read_text()
	hidden.hide_render = False
	layer_for(hidden).exclude = True
	export(root / "excluded-collection-light")
	assert "sunlight " not in (root / "excluded-collection-light" / "scene.xenon").read_text()
	layer_for(hidden).exclude = False
	export(root / "visible-collection-light")
	assert "sunlight " in (root / "visible-collection-light" / "scene.xenon").read_text()

bpy.utils.unregister_class(CaptureRender)
print(
	"Scatter visibility matches render evaluation; unsupported hair/density/children fail explicitly"
)
