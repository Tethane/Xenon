"""Independent end-to-end light/volume references, tested on each backend."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
backend = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="xenon-transport-") as temp:
	root = Path(temp)

	def render(text, samples=8, bounces=1):
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
				str(bounces),
				"--max-bounces",
				str(bounces),
				"--seed",
				"123",
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
		image = struct.unpack(("<" if float(scale) < 0 else ">") + f"{24 * 24 * 3}f", data)
		return [sum(image[c::3]) / (24 * 24) for c in range(3)]

	absorption = render(
		"camera 0 0 3 0 0 0 .001\nsky 1 1 1 1 1 1 1 1 1 1 1\n"
		"volume_box -100 -100 -1 100 100 1 .2 .5 1 0 0 0 0\n"
	)
	assert max(abs(a - math.exp(-2 * s)) for a, s in zip(absorption, [0.2, 0.5, 1])) < 1e-5, (
		absorption
	)
	# Integral of a unit directional source scattered once back toward the
	# camera through a two-unit slab, with extinction on both segments.
	scattering = render(
		"camera 0 0 3 0 0 0 .001\n"
		"volume_box -100 -100 -1 100 100 1 .2 .2 .2 .4 .4 .4 0\n"
		"sunlight 0 0 1 1 1 1 1\n",
		256,
	)
	reference = 0.4 / (4 * math.pi) * (1 - math.exp(-2 * 0.6 * 2)) / (2 * 0.6)
	assert max(abs(v - reference) for v in scattering) < 0.0006, (scattering, reference)
	(root / "matte.mat").write_text(
		"version 2\nname matte\nbaseColor .4 .4 .4\nior 1\nroughness 1\n"
	)
	surface = 'matfile "matte.mat"\nplane 0 0 1 0 matte\n'
	point = render(surface + f"pointlight 0 0 4 {16 * math.pi} {16 * math.pi} {16 * math.pi}\n")
	assert max(abs(v - 0.4) for v in point) < 1e-5, point
	shadow = render(
		surface + f"pointlight 0 0 4 {16 * math.pi} {16 * math.pi} {16 * math.pi}\n"
		"volume_box -100 -100 2 100 100 3 .3 .3 .3 0 0 0 0\n"
	)
	assert max(abs(v - 0.4 * math.exp(-0.3)) for v in shadow) < 1e-5, shadow
	spot = render(
		surface + f"spotlight 0 0 4 0 0 -1 {16 * math.pi} {16 * math.pi} {16 * math.pi} 10 20\n"
	)
	assert max(abs(v - 0.4) for v in spot) < 1e-5, spot
	dark = render(
		surface + f"spotlight 0 0 4 1 0 0 {16 * math.pi} {16 * math.pi} {16 * math.pi} 10 20\n"
	)
	assert max(dark) == 0, dark
	furnace = render(surface + "sky 1 1 1 1 1 1 1 1 1 1 1\n", 256)
	assert max(abs(v - 0.4) for v in furnace) < 0.004, furnace
	(root / "emitter.mat").write_text("version 2\nname emitter\nbaseColor 0 0 0\nemission 2 3 4\n")
	(root / "plane.obj").write_text("v -1 -1 0\nv 1 -1 0\nv 0 1 0\nusemtl emitter\nf 1 2 3\n")
	asset = 'matfile "emitter.mat"\nmesh "plane.obj"\n'
	omitted = render(asset + "instance 0 1 0 0 5 0 1 0 0 0 0 1 0\n")
	assert omitted == [0, 0, 0], "Explicit instances did not suppress automatic identity placement"
	shared = render(
		asset + "instance 0 2 .5 0 0 0 3 0 0 0 0 .5 0\ninstance 0 1 0 0 5 0 1 0 0 0 0 1 0\n"
	)
	assert shared == [2, 3, 4], ("Instanced emission / affine mesh placement failed", shared)
	(root / "glass.mat").write_text(
		"version 2\nname glass\nbaseColor 1 1 1\nior 1\nroughness 0\ntransmission 1\nabsorption .2 .5 1\n"
	)
	glass = 'matfile "glass.mat"\nsphere 0 0 0 1 glass\nsky 1 1 1 1 1 1 1 1 1 1 1\n'
	through = render("camera 0 0 3 0 0 0 .001\n" + glass, bounces=4)
	assert max(abs(a - math.exp(-2 * c)) for a, c in zip(through, [0.2, 0.5, 1])) < 0.0002, (
		"Closed-surface absorption failed",
		through,
	)
	inside = render("camera 0 0 0 0 0 -1 .001\ncamera_medium glass\n" + glass, bounces=2)
	assert max(abs(a - math.exp(-c)) for a, c in zip(inside, [0.2, 0.5, 1])) < 0.0002, (
		"Initial camera medium failed",
		inside,
	)
	print(
		f"{backend}: absorption={absorption}, single_scattering={scattering}, reference={reference}, furnace={furnace}"
	)
print(
	f"Analytic point/spot, homogeneous absorption/scattering, shadow transmittance and environment NEE passed ({backend})"
)
