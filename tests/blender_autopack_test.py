#!/usr/bin/env python3
"""A missing, unused source texture must not break saving a prepared scene."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from blender_export import arguments
from blender_prepare_scene import prepare

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.camera_add(location=(0, 0, 5))
bpy.context.scene.camera = bpy.context.object
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
mat = bpy.data.materials.new("Supported surface with unused source image")
mat.use_nodes = True
obj.data.materials.append(mat)
missing = bpy.data.images.new("Unused missing displacement", width=1, height=1)
missing.source = "FILE"
missing.filepath = "/nonexistent/xenon-unused-displacement.png"
missing.buffers_free()
mat.node_tree.nodes.new("ShaderNodeTexImage").image = missing
bpy.data.use_autopack = True
with tempfile.TemporaryDirectory(prefix="xenon_autopack_") as directory:
	root = Path(directory)
	args = arguments(["--output", str(root), "--instances", "--width", "16", "--height", "16"])
	report = prepare(args, SimpleNamespace(bake_size=16, bake_world=False, max_objects=0))
	assert report["status"] == "ok"
	assert (root / "comparison.blend").is_file()
	assert (root / "scene.xenon").is_file()
	assert not bpy.data.use_autopack
	assert any("Automatic packing disabled" in str(change) for change in report["changes"])
print("PASS: unused missing image does not block prepared scene saving or strict export")
