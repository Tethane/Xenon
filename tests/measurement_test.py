"""End-to-end validation of linear radiance, seeds, timing schema and outputs."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
backend = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="xenon-measurement-") as directory:
	root = Path(directory)
	scene = root / "test.xenon"
	scene.write_text("config 7 5 8\ncamera 0 0 1 0 0 0 90\nsky 1 2 4 1 2 4 1 2 4 1 1\n")
	base = [
		str(binary),
		"--scene",
		str(scene),
		"--backend",
		backend,
		"--no-display",
		"--output",
		str(root / "beauty.png"),
		"--linear-output",
		str(root / "linear.pfm"),
		"--metrics",
		str(root / "metrics.json"),
		"--seed",
		"18446744073709551615",
	]

	def run(extra=(), success=True):
		result = subprocess.run(base + list(extra), capture_output=True, text=True, timeout=45)
		assert (result.returncode == 0) == success, result.stdout + result.stderr
		return result

	run()
	linear = (root / "linear.pfm").read_bytes()
	magic, size, scale, data = linear.split(b"\n", 3)
	assert magic == b"PF" and size == b"7 5"
	values = struct.unpack(("<" if float(scale) < 0 else ">") + "105f", data)
	assert list(values) == [1.0, 2.0, 4.0] * 35, "PFM clipped HDR or applied display transform"
	metrics = json.loads((root / "metrics.json").read_text())
	assert metrics["backend"] == backend and metrics["samples"] == 8
	assert metrics["seed"] == 18446744073709551615
	assert metrics["max_radiance"] == 4 and metrics["device"]
	assert (
		metrics["seconds"]["render"] > 0
		and metrics["seconds"]["total"] >= metrics["seconds"]["render"]
	)
	assert all(v >= 0 for v in metrics["seconds"].values())
	run(["--exposure", "-2"])
	assert (root / "linear.pfm").read_bytes() == linear, "Exposure changed linear radiance"
	# Seed and orientation checks use a continuous directional environment.
	scene.write_text("config 7 5 8\ncamera 0 0 1 0 0 0 90\nsky 1 0 0 0 0 0 0 0 1 1 1\n")
	run(["--seed", "123"])
	one = (root / "linear.pfm").read_bytes()
	run(["--seed", "123"])
	assert (root / "linear.pfm").read_bytes() == one, "Same seed is not repeatable"
	run(["--seed", "124"])
	assert (root / "linear.pfm").read_bytes() != one, "Seed override ignored"
	run(["--seed", str(123 + (1 << 63))])
	assert (root / "linear.pfm").read_bytes() != one, "Seed high bit ignored"
	_, _, scale, data = one.split(b"\n", 3)
	values = struct.unpack(("<" if float(scale) < 0 else ">") + "105f", data)
	# PFM file rows are bottom-up; positive-y/red sky is at last row.
	assert values[2] > values[-1] and values[-3] > values[0], "PFM row order incorrect"
	run(["--width", "3", "--height", "2"])
	assert (root / "linear.pfm").read_bytes().split(b"\n")[1] == b"3 2"
	for args in (
		["--seed", "-1"],
		["--seed", "18446744073709551616"],
		["--seed", "12x"],
		["--width", "0"],
		["--height", "-1"],
		["--linear-output", ""],
		["--metrics", ""],
		["--linear-output", str(root / "beauty.png")],
		["--metrics", str(root / "linear.pfm")],
	):
		run(args, success=False)
print(f"Linear HDR, seeded sampling, and measurement checks passed ({backend})")
