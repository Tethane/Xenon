import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

binary = Path(sys.argv[1]).resolve()
backend = sys.argv[2]
cuda_built = sys.argv[3] == "1"
env = os.environ.copy()
env.pop("DISPLAY", None)
env.pop("WAYLAND_DISPLAY", None)


def png_pixels(path):
	data = path.read_bytes()
	assert data[:8] == b"\x89PNG\r\n\x1a\n"
	offset, packed = 8, b""
	while offset < len(data):
		size = struct.unpack_from(">I", data, offset)[0]
		kind = data[offset + 4 : offset + 8]
		chunk = data[offset + 8 : offset + 8 + size]
		if kind == b"IHDR":
			width, height, depth, color = struct.unpack_from(">IIBB", chunk)
			assert (width, height, depth, color) == (7, 5, 8, 2)
		elif kind == b"IDAT":
			packed += chunk
		offset += size + 12
	raw = zlib.decompress(packed)
	stride, pixels, previous = width * 3, bytearray(), bytearray(width * 3)
	for y in range(height):
		start = y * (stride + 1)
		filter_type = raw[start]
		row = bytearray(raw[start + 1 : start + 1 + stride])
		for x in range(stride):
			a = row[x - 3] if x >= 3 else 0
			b = previous[x]
			c = previous[x - 3] if x >= 3 else 0
			prediction = a + b - c
			distances = [abs(prediction - v) for v in (a, b, c)]
			paeth = (a, b, c)[distances.index(min(distances))]
			row[x] = (row[x] + (0, a, b, (a + b) // 2, paeth)[filter_type]) & 255
		pixels.extend(row)
		previous = row
	return pixels


with tempfile.TemporaryDirectory(prefix="xenon-execution-") as directory:
	root = Path(directory)
	scene_dir = root / "scene with spaces"
	scene_dir.mkdir()
	(scene_dir / "materials").mkdir()
	material = scene_dir / "materials/matte.mat"
	material.write_text("name matte\nbaseColor 0.7 0.6 0.5\nroughness 1\n")
	mesh = scene_dir / "test mesh.obj"
	mesh.write_text(
		"v -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nusemtl matte\nf 1 2 3\nf 1 3 4\n"
	)
	scene = scene_dir / "test.xenon"
	content = (
		"config 7 5 3\ncamera 0 0 3 0 0 0 40\nmatfile matte.mat\n"
		'mesh "test mesh.obj"\nsky 0.8 0.7 0.6 0.8 0.7 0.6 0.8 0.7 0.6 1 3\n'
	)
	scene.write_text(content)
	output = root / "nested/render.png"
	base = [
		str(binary),
		"--scene",
		str(scene),
		"--no-display",
		"--backend",
		backend,
		"--min-bounces",
		"0",
		"--max-bounces",
		"2",
		"--output",
		str(output),
	]

	def run(extra=(), success=True, message=None, command=None, environment=env):
		result = subprocess.run(
			command or base + list(extra),
			cwd=root,
			env=environment,
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			timeout=30,
		)
		assert (result.returncode == 0) == success, result.stdout
		if message:
			assert message in result.stdout, result.stdout
		return result

	run(message="Progress: 3/3 samples")
	expected = output.read_bytes()
	assert any(png_pixels(output)), "Indirect light missing at two bounces"
	run()
	assert output.read_bytes() == expected, "Headless render is not repeatable"
	run(["--max-bounces", "1"])
	# Environment NEE now provides direct illumination at the first surface.
	assert any(png_pixels(output)), "One-bounce environment NEE is missing"
	assert output.read_bytes() != expected, "max-bounces=1 was ignored"
	run(["--samples", "1"], message="Progress: 1/1 samples")
	one_sample = output.read_bytes()
	run(["--samples", "2"], message="Progress: 2/2 samples")
	assert output.read_bytes() != one_sample, "Sample override was ignored"
	run(["--samples", "3"])
	assert output.read_bytes() == expected, "Final sample was not saved"

	for args, message in [
		(["--samples", "0"], "Samples must be positive"),
		(["--samples", "2junk"], "Invalid integer"),
		(["--samples", "999999999999999999999"], "Invalid integer"),
		(["--min-bounces", "-1"], "Bounces must satisfy"),
		(["--min-bounces", "3"], "Bounces must satisfy"),
		(["--max-bounces", "0"], "Bounces must satisfy"),
		(["--backend", "typo"], "Unknown backend"),
		(["--samples"], "Missing value"),
		(["--unknown"], "Unknown argument"),
		(["--exposure", "nan"], "Exposure must be finite"),
		(["--exposure", "inf"], "Exposure must be finite"),
		(["--exposure", "33"], "Exposure must be finite"),
		(["--exposure", "2junk"], "Invalid number"),
		(["--output", ""], "Output filename"),
		(["--output", str(root)], "Is a directory"),
		(["--output", str(output / "child.png")], "Not a directory"),
		(["--output", "/dev/full"], "Failed to write PNG"),
	]:
		run(args, success=False, message=message)

	if not cuda_built:
		run(["--backend", "cuda"], success=False, message="not available in this build")
	if backend == "cuda":
		no_device = dict(env, CUDA_VISIBLE_DEVICES="")
		run(success=False, message="CUDA backend is unavailable", environment=no_device)
		scene.write_text(content + "sphere 0 0 0 1 matte\n")
		run()
		scene.write_text(content)
	else:
		command = [arg for arg in base if arg != "--no-display"]
		run(command=command, success=False, message="Could not initialize preview")

	for broken, message in [
		(content.replace('"test mesh.obj"', '"missing.obj"'), "Failed to load mesh"),
		(content.replace("matte.mat", "missing.mat"), "Failed to load material"),
		(content + "sphere 0 0 0 1 unknown\n", "Unknown material"),
		(content + "light 1 0 1 1 1 1\n", "Invalid light reference"),
		(content + "light 0 2 1 1 1 1\n", "Invalid light reference"),
		(content + "light 0 0 1 1 1 0\n", "Invalid light reference"),
		(content + "config 0 5 3\n", "Invalid image dimensions"),
		(content + "config 2147483647 5 3\n", "Invalid image dimensions"),
		(content + "config 7 5\n", "Invalid fields"),
		(content + "camera 0 0 0 0 0 0 40\n", "Invalid camera"),
		(content + "sphere 0 0 0 0 matte\n", "Sphere radius"),
		(content + "box 0 0 0 1 0 1 matte\n", "Box extents"),
		(content + "disk 0 0 0 0 0 0 1 matte\n", "Invalid disk"),
		(content + "quad 0 0 0 1 0 0 2 0 0 matte\n", "Degenerate quad"),
		(content + "plane 0 0 0 1 matte\n", "Invalid plane normal"),
		(content + "sunlight 0 0 0 1 1 1 1\n", "Invalid sunlight"),
		(content + "sky 1 1 1 1 1 1 1 1 1 1 0\n", "Sky sharpness"),
		(content + "unknown_command 1\n", "Unknown scene command"),
		(content.replace('"test mesh.obj"', '"test mesh.obj" 0 0'), "Invalid fields"),
	]:
		scene.write_text(broken)
		run(success=False, message=message)
	scene.write_text(content)
	scene.write_text("config 7 5 16\ncamera 0 0 1 0 0 0 90\nsky 1 0 0 0 0 0 0 0 1 1 1\n")
	run()
	pixels = png_pixels(output)
	top, bottom = 3 * 3, (4 * 7 + 3) * 3
	assert pixels[top] > pixels[bottom] and pixels[bottom + 2] > pixels[top + 2], (
		"Saved PNG is vertically inverted"
	)
	scene.write_text(content)
	scene.write_text(
		"config 7 5 3\ncamera 0 0 1.5 0 0 0 40\n"
		"material test 0.5 0.5 0.5 0 1 0.5 1\nplane 0 0 2 2 test\n"
		"sky 0 0 0 0 0 0 0 0 0 0 1\nsunlight 0 0 1 1 1 1 1\n"
	)
	run()
	assert any(png_pixels(output)), "Plane normalization changed the plane offset"
	scene.write_text(content)
	scene.write_text("config 7 5 1\ncamera 0 0 1 0 0 0 40\nsky 1 2 4 1 2 4 1 2 4 1 1\n")
	run()
	assert list(png_pixels(output)) == [188, 213, 231] * 35, (
		"HDR output clipped or incorrectly encoded"
	)
	run(["--samples", "16"])
	assert list(png_pixels(output)) == [188, 213, 231] * 35, "Brightness grows with sample count"
	run(["--exposure", "-1"])
	assert list(png_pixels(output)) == [156, 188, 213] * 35, (
		"Exposure not applied before tone mapping"
	)
	scene.write_text(content)
	material.write_text("name matte\nbaseColor 1\n")
	run(success=False, message="Failed to load material")
	material.write_text("name matte\n")
	mesh.write_text(mesh.read_text().replace("usemtl matte", "usemtl unknown"))
	run(success=False, message="unknown material")
	mesh.write_text("")
	run(success=False, message="Failed to read triangles")
	scene.unlink()
	run(success=False, message="Failed to open scene")

print(f"Execution and scene validation passed ({backend})")
