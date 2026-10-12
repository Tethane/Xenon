#!/usr/bin/env python3
"""Repeated static placements reuse mesh extraction; overrides/deformation remain distinct."""

from pathlib import Path
import sys
import tempfile
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import Exporter, arguments

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.camera_add(location=(0, 0, 10))
bpy.context.scene.camera = bpy.context.object
bpy.ops.mesh.primitive_cube_add()
base = bpy.context.object
red = bpy.data.materials.new("Red")
red.use_nodes = True
red.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)
blue = red.copy()
blue.name = "Blue"
blue.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 0, 1, 1)
base.data.materials.append(red)
for i in range(100):
	obj = bpy.data.objects.new(f"Placement {i}", base.data)
	bpy.context.scene.collection.objects.link(obj)
	obj.location.x = i * 3
other = bpy.data.objects.new("Material override", base.data)
bpy.context.scene.collection.objects.link(other)
other.material_slots[0].link = "OBJECT"
other.material_slots[0].material = blue
changed = bpy.data.objects.new("Deformed", base.data)
bpy.context.scene.collection.objects.link(changed)
modifier = changed.modifiers.new("Displacement", "DISPLACE")
modifier.direction = "Z"
modifier.strength = 0.5
bpy.context.view_layer.update()
with tempfile.TemporaryDirectory(prefix="xenon_mesh_cache_") as directory:
	exporter = Exporter(arguments(["--output", directory, "--instances"]))
	exporter.run()
	report = exporter.report
	assert report["status"] == "ok"
	assert report["mesh_placements"] == 103
	assert report["mesh_assets"] == 3, report["mesh_assets"]
	assert report["mesh_data_evaluations"] == 3, report["mesh_data_evaluations"]
	by_name = {o["name"]: o for o in report["objects"]}
	asset = by_name[base.name]["mesh_asset"]
	assert all(by_name[f"Placement {i}"]["mesh_asset"] == asset for i in range(100))
	assert by_name["Material override"]["mesh_asset"] != asset
	assert by_name["Deformed"]["mesh_asset"] != asset
print(
	"PASS: 103 placements, 3 mesh extractions; material overrides and object deformation remain separate"
)
