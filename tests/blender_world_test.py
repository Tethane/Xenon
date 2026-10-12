#!/usr/bin/env python3
"""Blender integration regression for environment orientation and diffuse branch baking.

Run: blender --background --factory-startup --disable-autoexec --python-exit-code 1
     --python tests/blender_world_test.py
"""

import json, sys
from pathlib import Path
import bpy, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_prepare_world import prepare_world

w, h = 64, 32
pixels = np.zeros((h, w, 4), dtype=np.float32)
pixels[:, :, 0] = (np.arange(w) + 0.5)[None, :] / w
pixels[:, :, 1] = (np.arange(h) + 0.5)[:, None] / h
pixels[:, :, 2] = 2.0
pixels[:, :, 3] = 1.0
im = bpy.data.images.new(
	"Known linear directional environment", width=w, height=h, float_buffer=True
)
im.colorspace_settings.name = "Non-Color"
im.pixels.foreach_set(pixels.ravel())
world = bpy.data.worlds.new("Known gradient world")
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
out = nodes.new("ShaderNodeOutputWorld")
bg = nodes.new("ShaderNodeBackground")
tex = nodes.new("ShaderNodeTexEnvironment")
tex.image = im
world.node_tree.links.new(tex.outputs["Color"], bg.inputs["Color"])
world.node_tree.links.new(bg.outputs[0], out.inputs["Surface"])
scene = bpy.context.scene
scene.world = world
r = prepare_world(scene, "build/world-orientation", width=w, height=h, samples=64)
result = bpy.data.images.load(r["path"], check_existing=False)
result.colorspace_settings.name = "Non-Color"
actual = np.array(result.pixels[:], dtype=np.float32).reshape(h, w, 4)
print(
	"WORLD_PROBE",
	json.dumps(
		{
			"mae": float(np.abs(actual[2:-2, 2:-2, :3] - pixels[2:-2, 2:-2, :3]).mean()),
			"points": [
				[list(map(float, actual[y, x, :3])), list(map(float, pixels[y, x, :3]))]
				for x, y in [(16, 8), (32, 16), (48, 24)]
			],
		}
	),
)
assert np.abs(actual[2:-2, 2:-2, :3] - pixels[2:-2, 2:-2, :3]).mean() < 0.02, (
	"world orientation or linear radiance mismatch"
)
assert scene.world != world and world.node_tree.nodes.get("Environment Texture").image == im
# The derived world must use the diffuse lighting branch while preserving its
# source's independent camera branch and HDR intensity.
world2 = bpy.data.worlds.new("Ray-dependent source")
world2.use_nodes = True
tree = world2.node_tree
tree.nodes.clear()
out = tree.nodes.new("ShaderNodeOutputWorld")
mix = tree.nodes.new("ShaderNodeMixShader")
light = tree.nodes.new("ShaderNodeLightPath")
diffuse = tree.nodes.new("ShaderNodeBackground")
diffuse.inputs["Color"].default_value = (3, 0, 0, 1)
camera = tree.nodes.new("ShaderNodeBackground")
camera.inputs["Color"].default_value = (0, 0, 1, 1)
tree.links.new(light.outputs["Is Camera Ray"], mix.inputs[0])
tree.links.new(diffuse.outputs[0], mix.inputs[1])
tree.links.new(camera.outputs[0], mix.inputs[2])
tree.links.new(mix.outputs[0], out.inputs["Surface"])
scene.world = world2
report = prepare_world(scene, "build/world-diffuse-branch", width=8, height=4, samples=1)
assert report["light_path_changes"][0]["output"] == "Is Camera Ray"
result = bpy.data.images.load(report["path"], check_existing=False)
values = np.array(result.pixels[:]).reshape(-1, 4)
assert np.allclose(values[:, :3], (3, 0, 0), atol=0.02)
assert light.outputs["Is Camera Ray"].is_linked and scene.world != world2
# A participating world medium is outside the environment-image model.
scene.world = world2
volume = tree.nodes.new("ShaderNodeVolumeScatter")
tree.links.new(volume.outputs[0], out.inputs["Volume"])
try:
	prepare_world(scene, "build/world-rejected-volume", width=8, height=4, samples=1)
except ValueError as error:
	assert "World volumes" in str(error)
else:
	raise AssertionError("World volume was silently discarded")
assert scene.world == world2
print("WORLD_ORIENTATION_DIFFUSE_BRANCH_AND_VOLUME_REJECTION_OK")
