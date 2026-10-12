"""Front/back emission and area-light MIS against independent radiance integrals."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
backend = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="xenon-two-sided-") as temp:
	root = Path(temp)

	def emitter(two_sided):
		(root / "emitter.mat").write_text(
			"version 2\nname emitter\nbaseColor 0 0 0\nior 1\nroughness 1\n"
			f"emission 2 3 4\nemissionTwoSided {two_sided}\n"
		)

	def mesh(height, front_down=False):
		# Default face normal is +Z; front_down reverses both triangles.
		faces = "f 1 3 2\nf 1 4 3\n" if front_down else "f 1 2 3\nf 1 3 4\n"
		(root / "emitter.obj").write_text(
			f"v -1 -1 {height}\nv 1 -1 {height}\nv 1 1 {height}\nv -1 1 {height}\nusemtl emitter\n"
			+ faces
		)

	def render(text, samples=8, bounces=1):
		(root / "scene.xenon").write_text(
			"version 2\nconfig 24 24 8\ncamera 0 0 1 0 0 0 .001\n"
			'sky 0 0 0 0 0 0 0 0 0 0 1\nmatfile "emitter.mat"\n' + text
		)
		result = subprocess.run(
			[
				str(binary),
				"--scene",
				str(root / "scene.xenon"),
				"--backend",
				backend,
				"--no-display",
				"--samples",
				str(samples),
				"--min-bounces",
				"0",
				"--max-bounces",
				str(bounces),
				"--seed",
				"2459",
				"--output",
				str(root / "image.png"),
				"--linear-output",
				str(root / "image.pfm"),
			],
			text=True,
			capture_output=True,
			timeout=120,
		)
		assert result.returncode == 0, result.stdout + result.stderr
		magic, size, scale, data = (root / "image.pfm").read_bytes().split(b"\n", 3)
		pixels = struct.unpack(("<" if float(scale) < 0 else ">") + f"{24 * 24 * 3}f", data)
		assert all(math.isfinite(v) for v in pixels)
		return [sum(pixels[c::3]) / (24 * 24) for c in range(3)]

	def close(actual, expected, tolerance, label):
		assert max(abs(a - e) for a, e in zip(actual, expected)) < tolerance, (
			label,
			actual,
			expected,
		)

	emitter(0)
	mesh(0)
	shape = 'mesh "emitter.obj"\n'
	close(render(shape), [2, 3, 4], 1e-6, "default front-facing mesh emission")
	backside = "camera 0 0 -1 0 0 0 .001\n"
	close(render(backside + shape), [0, 0, 0], 1e-6, "default back-facing mesh is dark")
	emitter(1)
	close(render(shape), [2, 3, 4], 1e-6, "two-sided front radiance is unchanged")
	close(render(backside + shape), [2, 3, 4], 1e-6, "two-sided back radiance is equal")
	close(
		render(backside + "plane 0 0 1 0 emitter\n"),
		[2, 3, 4],
		1e-6,
		"analytic primitive two-sided back radiance",
	)
	(root / "matte.mat").write_text(
		"version 2\nname matte\nbaseColor .4 .4 .4\nroughness 1\nior 1\n"
	)
	receiver = 'matfile "matte.mat"\nplane 0 0 1 0 matte\n'
	mesh(2)
	# Emitter faces away from the receiver. Enabling two-sided emission must
	# illuminate the receiver without changing geometry, camera or intensity.
	direct = render(receiver + shape, 512)
	mis = render(receiver + shape, 1024, 2)
	grid = 400
	integral = 0.0
	for y in range(grid):
		yy = -1 + 2 * (y + 0.5) / grid
		for x in range(grid):
			xx = -1 + 2 * (x + 0.5) / grid
			integral += 4.0 / (xx * xx + yy * yy + 4.0) ** 2
	form_factor = 0.4 / math.pi * integral * 4 / (grid * grid)
	reference = [form_factor * v for v in [2, 3, 4]]
	close(direct, reference, 0.0015, "back-facing two-sided NEE against independent area integral")
	close(mis, reference, 0.0015, "two-sided NEE and BSDF-hit MIS preserve total radiance")
	emitter(0)
	close(
		render(receiver + shape, 16, 2),
		[0, 0, 0],
		1e-6,
		"default back emitter remains dark through NEE and MIS",
	)
	mesh(2, True)
	front = render(receiver + shape, 512)
	close(front, reference, 0.0015, "flipped one-sided emitter matches two-sided backside")
	close(
		front, direct, 0.0025, "front versus backside direct lighting agrees within sampling error"
	)
	emitter(2)
	(root / "scene.xenon").write_text('version 2\nmatfile "emitter.mat"\n')
	rejected = subprocess.run(
		[str(binary), "--scene", str(root / "scene.xenon"), "--backend", backend, "--no-display"],
		capture_output=True,
		text=True,
		timeout=30,
	)
	assert rejected.returncode != 0 and "emissionTwoSided" in rejected.stdout + rejected.stderr, (
		"invalid boolean accepted"
	)
	print(f"{backend}: two-sided direct={direct}, MIS={mis}, independent reference={reference}")
