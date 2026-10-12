#!/usr/bin/env python3
"""Test explicit preparation omissions using Blender's real render depsgraph.

blender --background --factory-startup --threads 2 --python-exit-code 1 \
  --python tests/blender_omissions_test.py
No Cycles shading, GPU work, or full-scene import is performed.
"""

from collections import Counter
import hashlib
from pathlib import Path
import sys
import tempfile

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import Exporter, arguments
from blender_prepare_scene import capture_render


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.world = None
scene.render.engine = "CYCLES"
scene.render.resolution_x = 32
scene.render.resolution_y = 18
scene.render.resolution_percentage = 50
scene.render.use_border = True
bpy.ops.object.camera_add(location=(0, 0, 5))
camera = bpy.context.object
camera.name = "Active camera"
scene.camera = camera

material = bpy.data.materials.new("Retained material")
material.use_nodes = True
mesh = bpy.data.meshes.new("Triangle")
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
mesh.materials.append(material)
kept = bpy.data.objects.new("Window Glass.001", mesh)
scene.collection.objects.link(kept)
kept.location = (-2, 0, 0)

# The source prototype is excluded from direct rendering but still participates
# through two separately transformed collection instances.
prototypes = bpy.data.collections.new("Prototype source")
scene.collection.children.link(prototypes)
prototype = bpy.data.objects.new("Window Glass", mesh)
prototypes.objects.link(prototype)
bpy.context.view_layer.layer_collection.children[prototypes.name].exclude = True
for index in range(2):
	instancer = bpy.data.objects.new(f"Placement {index}", None)
	instancer.instance_type = "COLLECTION"
	instancer.instance_collection = prototypes
	instancer.location = (index * 2, 0, 0)
	scene.collection.objects.link(instancer)

for name, kind in [
	("Area key", "AREA"),
	("Area fill", "AREA"),
	("Key point", "POINT"),
	("Sun", "SUN"),
]:
	data = bpy.data.lights.new(name, kind)
	light = bpy.data.objects.new(name, data)
	scene.collection.objects.link(light)
	light.location = (0, 0, 3)
bpy.context.view_layer.update()


def source_state():
	settings = {
		name: getattr(scene.render, name)
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
	objects = {
		obj.name: (
			obj.hide_render,
			tuple(value for row in obj.matrix_world for value in row),
			tuple(slot.material.name if slot.material else None for slot in obj.material_slots),
		)
		for obj in scene.objects
	}
	return settings, objects, scene.camera.name


def geometry(result):
	return Counter((instance["name"], instance["instance"]) for instance in result["instances"])


def lights(result):
	return {name: data.type for name, data, _ in result["lights"]}


with tempfile.TemporaryDirectory(prefix="xenon_omissions_") as temporary:
	root = Path(temporary)
	source = root / "source.blend"
	bpy.ops.wm.save_as_mainfile(filepath=str(source))
	checksum = hashlib.sha256(source.read_bytes()).hexdigest()
	before = source_state()

	def capture(name, **options):
		validator = Exporter(
			arguments(
				["--output", str(root / name), "--width", "4", "--height", "4", "--samples", "1"]
			)
		)
		result = capture_render(scene, validator, **options)
		assert validator.report["depsgraph_mode"] == "RENDER"
		assert source_state() == before, "Capture changed source objects or render settings"
		assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum, (
			"Capture changed original file bytes"
		)
		return result

	expected_geometry = Counter({("Window Glass", True): 2, ("Window Glass.001", False): 1})
	expected_lights = {"Area key": "AREA", "Area fill": "AREA", "Key point": "POINT", "Sun": "SUN"}
	baseline = capture("baseline")
	assert geometry(baseline) == expected_geometry, geometry(baseline)
	assert lights(baseline) == expected_lights, lights(baseline)
	assert baseline["omitted"] == {}
	placements = [
		tuple(instance["matrix"].translation)
		for instance in baseline["instances"]
		if instance["name"] == "Window Glass"
	]
	assert set(placements) == {(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)}, placements

	selected = capture("selected", omit_objects=("Window Glass",))
	assert geometry(selected) == Counter({("Window Glass.001", False): 1}), geometry(selected)
	assert lights(selected) == expected_lights
	assert selected["omitted"] == {"Window Glass": 2}, selected["omitted"]
	assert selected["camera"] is not None

	areas = capture("areas", omit_area_lights=True)
	assert geometry(areas) == expected_geometry
	assert lights(areas) == {"Key point": "POINT", "Sun": "SUN"}
	assert areas["omitted"] == {"Area key": 1, "Area fill": 1}, areas["omitted"]

	combined = capture(
		"combined", omit_objects=("Window Glass", "Key point"), omit_area_lights=True
	)
	assert geometry(combined) == Counter({("Window Glass.001", False): 1})
	assert lights(combined) == {"Sun": "SUN"}
	assert combined["omitted"] == {"Window Glass": 2, "Key point": 1, "Area key": 1, "Area fill": 1}

	for name, requested, message in [
		("typo", ("Window glass",), "not found"),
		("camera", (camera.name,), "active camera"),
	]:
		try:
			capture(name, omit_objects=requested)
		except RuntimeError as error:
			assert message in str(error), (message, str(error))
		else:
			raise AssertionError(f"{name}: invalid omission was accepted")
		assert source_state() == before
		assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum

print(
	"PASS exact-name omissions remove both placements; other geometry/point/sun remain; area omission is opt-in; typo/camera reject; source bytes/settings unchanged"
)
