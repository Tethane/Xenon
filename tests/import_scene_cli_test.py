"""Host-side import CLI checks; no Blender installation or rendering required."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/import_scene.py"
spec = importlib.util.spec_from_file_location("import_scene", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="xenon-import-cli-") as directory:
	root = Path(directory)
	source = root / "source ; literal$.blend"
	source.write_bytes(b"original Blender source bytes")
	original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
	capture = root / "command.json"
	blender = root / "mock blender"
	blender.write_text(
		f"#!{sys.executable}\n"
		+ """import json,os,sys
from pathlib import Path
a=sys.argv[1:]
Path(os.environ['MOCK_COMMAND']).write_text(json.dumps(a))
args=a[a.index('--')+1:];out=Path(args[args.index('--output')+1])
out.mkdir(parents=True,exist_ok=True)
if os.environ.get('MOCK_FAIL'):
 (out/'export-report.json').write_text(json.dumps({'status':'unsupported','errors':['unsupported test graph']}))
 sys.exit(7)
(out/'export-report.json').write_text(json.dumps({'status':'ok','errors':[]}))
(out/'scene.xenon').write_text('version 2\\nconfig 32 24 1\\n')
"""
	)
	blender.chmod(0o755)
	env = {**os.environ, "MOCK_COMMAND": str(capture), "BLENDER": str(blender)}

	def run(*args, success=True, extra=None):
		result = subprocess.run(
			[sys.executable, str(SCRIPT), *map(str, args)],
			env={**env, **(extra or {})},
			capture_output=True,
			text=True,
		)
		assert (result.returncode == 0) == success, result.stdout + result.stderr
		return result

	assert "replaces imported" in run("--help", extra={"BLENDER": "missing-blender"}).stdout
	assert not capture.exists()
	for args in [
		(root / "missing.blend", "--output", root / "missing"),
		(source, "--output", source),
		(source, "--output", root),
		(source, "--output", root / "bad", "--width", "0"),
		(source, "--output", root / "bad", "--samples", "-1"),
		(source, "--output", root / "bad", "--samples", "2147483648"),
		(source, "--output", root / "bad", "--width", "100000001"),
		(source, "--output", root / "bad", "--bake-size", "32"),
		(source, "--output", root / "bad", "--bake-world"),
		(source, "--output", root / "bad", "--volume-proxies"),
		(source, "--output", root / "bad", "--max-texture-size", "1024"),
		(source, "--output", root / "bad", "--prepare", "--max-texture-size", "0"),
		(source, "--output", root / "bad", "--omit-area-lights"),
		(source, "--output", root / "bad", "--omit-object", "Window Glass"),
		(source, "--output", root / "bad", "--prepare", "--omit-object", ""),
		(source, "--output", root / "bad", "--prepare", "--omit-object", "   "),
		(source, "--output", root / "bad", "--blender", "missing-blender"),
		(source, "--output", root / "bad", "--width", "10001", "--height", "10000"),
	]:
		run(*args, success=False)
		assert not capture.exists(), "Invalid input must not launch Blender"
	strict = root / "strict output"
	result = run(
		source,
		"--output",
		strict,
		"--width",
		"48",
		"--height",
		"32",
		"--samples",
		"3",
		"--threads",
		"2",
	)
	command = json.loads(capture.read_text())
	assert command.index("--disable-autoexec") < command.index(str(source))
	assert command[command.index("--python") + 1].endswith("/blender_export.py")
	assert command[command.index("--python-exit-code") + 1] == "1"
	assert command[command.index("--threads") + 1] == "2"
	assert (
		"--instances" in command
		and "--compensated-materials" not in command
		and "--bake-world" not in command
		and "--volume-proxies" not in command
	)
	assert "--max-texture-size" not in command
	assert "--omit-area-lights" not in command and not any(
		arg.startswith("--omit-object") for arg in command
	)
	assert (
		command[command.index("--width") + 1] == "48"
		and command[command.index("--samples") + 1] == "3"
	)
	request = json.loads((strict / "import-request.json").read_text())
	assert request["status"] == "complete" and request["source_sha256_after"] == original_hash
	assert request["omit_area_lights"] is False and request["omit_objects"] == []
	assert "./xenon.sh --scene" in result.stdout
	before = capture.read_bytes()
	run(source, "--output", strict, success=False)
	assert capture.read_bytes() == before, "Nonempty output must not be overwritten"
	prepared = root / "prepared"
	run(
		source,
		"--output",
		prepared,
		"--prepare",
		"--bake-size",
		"128",
		"--bake-world",
		"--volume-proxies",
		"--max-texture-size",
		"512",
		"--no-instances",
		"--compensated-materials",
		"--omit-area-lights",
		"--omit-object",
		"Window Glass",
		"--omit-object=--Panel $literal",
		"--blender",
		blender,
		extra={"BLENDER": "invalid-env-ignored-by-explicit-flag"},
	)
	command = json.loads(capture.read_text())
	assert command[command.index("--python") + 1].endswith("/blender_prepare_scene.py")
	assert "--instances" not in command and "--compensated-materials" in command
	assert command[command.index("--bake-size") + 1] == "128"
	assert "--bake-world" in command
	assert "--volume-proxies" in command
	assert command[command.index("--max-texture-size") + 1] == "512"
	assert json.loads((prepared / "import-request.json").read_text())["max_texture_size"] == 512
	assert "--omit-area-lights" in command
	assert [arg for arg in command if arg.startswith("--omit-object=")] == [
		"--omit-object=Window Glass",
		"--omit-object=--Panel $literal",
	]
	request = json.loads((prepared / "import-request.json").read_text())
	assert request["omit_area_lights"] is True and request["omit_objects"] == [
		"Window Glass",
		"--Panel $literal",
	]
	for extension in (".obj", ".gltf", ".glb"):
		model = root / ("model" + extension)
		model.write_text("mock source")
		output = root / ("export" + extension)
		result = run(model, "--output", output, "--max-texture-size", "1024")
		command = json.loads(capture.read_text())
		assert command[command.index("--python") + 1].endswith("/blender_import_scene.py")
		assert command[command.index("--input") + 1] == str(model)
		assert command[command.index("--max-texture-size") + 1] == "1024"
		assert "replaces imported lights" in result.stdout
		before = capture.read_bytes()
		run(model, "--output", root / "invalid-prepare", "--prepare", success=False)
		assert capture.read_bytes() == before
		for options in [("--omit-area-lights",), ("--omit-object", "Window Glass")]:
			run(model, "--output", root / "invalid-omit", *options, success=False)
			assert capture.read_bytes() == before
	failed = root / "failed"
	result = run(source, "--output", failed, success=False, extra={"MOCK_FAIL": "1"})
	assert "No fallback conversion" in result.stderr
	request = json.loads((failed / "import-request.json").read_text())
	assert request["status"] == "failed" and request["blender_exit_code"] == 7
	assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
	assert not (failed / "scene.xenon").exists()

	# Local discovery is numeric (5.2.10 after 5.2.2), and explicit settings win.
	for version in ("4.5.3", "5.2.2", "5.2.10"):
		path = root / f"build/tools/blender-{version}-linux-x64/blender"
		path.parent.mkdir(parents=True)
		path.write_text("#!/bin/sh\nexit 0\n")
		path.chmod(0o755)
	with patch.object(module, "ROOT", root), patch.dict(os.environ, {}, clear=True):
		assert "blender-5.2.10-" in str(module.find_blender())
		assert module.find_blender(str(blender)) == blender
		with patch.dict(os.environ, {"BLENDER": str(blender)}):
			assert module.find_blender() == blender

print(
	"Import CLI validation, strict/prepare/model delegation, discovery, provenance and no-overwrite checks passed."
)
