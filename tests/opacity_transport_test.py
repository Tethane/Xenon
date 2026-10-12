"""Independent coverage and unequal-emitter radiance references for each backend."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
backend = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="xenon-coverage-") as temp:
	root = Path(temp)

	def render(text, samples=256):
		(root / "scene.xenon").write_text(
			"version 2\nconfig 24 24 8\ncamera 0 0 1 0 0 0 .001\nsky 0 0 0 0 0 0 0 0 0 0 1\n" + text
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
				"1",
				"--seed",
				"9351",
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

	for name, alpha in [("a", 0.3), ("b", 0.6)]:
		(root / f"{name}.mat").write_text(
			f"version 2\nname {name}\nbaseColor 0 0 0\nior 1\nopacityMode blend\nopacity {alpha}\n"
		)
	mats = 'matfile "a.mat"\nmatfile "b.mat"\n'
	through = render(mats + "plane 0 0 1 0 a\nplane 0 0 1 -1 b\nsky 1 1 1 1 1 1 1 1 1 1 1\n")
	assert max(abs(v - 0.7 * 0.4) for v in through) < 0.006, through
	(root / "matte.mat").write_text(
		"version 2\nname matte\nbaseColor .4 .4 .4\nroughness 1\nior 1\n"
	)
	receiver = 'matfile "matte.mat"\nplane 0 0 1 0 matte\n'
	shadow = render(
		receiver + mats + f"plane 0 0 1 2 a\nplane 0 0 1 3 b\nsunlight 0 0 1 1 1 1 {math.pi}\n"
	)
	assert max(abs(v - 0.4 * 0.7 * 0.4) for v in shadow) < 0.003, shadow
	(root / "emitter.mat").write_text(
		"version 2\nname emitter\nbaseColor 0 0 0\nior 1\nopacityMode blend\nopacity .3\nemission 2 3 4\n"
	)
	emission = render('matfile "emitter.mat"\nplane 0 0 1 0 emitter\n')
	assert max(abs(v - e * 0.3) for v, e in zip(emission, [2, 3, 4])) < 0.012, emission
	# Two disjoint downward-facing square emitters. Numerically integrate their
	# geometric transport independently of triangle sampling and Xenon's PMFs.
	patches = [(0.5, 1.5, 1.0, 1.0), (-2.0, -1.0, 10.0, 0.35)]
	geometry = []
	for i, (xmin, xmax, radiance, alpha) in enumerate(patches):
		name = f"patch{i}"
		(root / f"{name}.mat").write_text(
			f"version 2\nname {name}\nbaseColor 0 0 0\nior 1\nemission {radiance} {radiance} {radiance}\nopacityMode blend\nopacity {alpha}\n"
		)
		# Clockwise in world XY: emitting front face points toward receiver.
		(root / f"{name}.obj").write_text(
			f"v {xmin} -.5 2\nv {xmin} .5 2\nv {xmax} .5 2\nv {xmax} -.5 2\nusemtl {name}\nf 1 2 3\nf 1 3 4\n"
		)
		geometry.append(f'matfile "{name}.mat"\nmesh "{name}.obj"\n')
	direct = render(receiver + "".join(geometry), 512)
	reference = 0.0
	grid = 400
	for xmin, xmax, radiance, alpha in patches:
		integral = 0.0
		for y in range(grid):
			yy = -0.5 + (y + 0.5) / grid
			for x in range(grid):
				xx = xmin + (x + 0.5) * (xmax - xmin) / grid
				integral += 4.0 / (xx * xx + yy * yy + 4.0) ** 2
		reference += 0.4 / math.pi * radiance * alpha * integral * (xmax - xmin) / (grid * grid)
	assert max(abs(v - reference) for v in direct) < 0.001, (direct, reference)
	print(
		f"{backend}: coverage={through}; shadow={shadow}; emitter={emission}; unequal-emitter NEE={direct}, independent reference={reference}"
	)
