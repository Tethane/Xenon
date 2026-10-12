#!/usr/bin/env python3
"""Restore the original scene meshes from a local Xenon checkout and verify them."""

import argparse
import hashlib
from pathlib import Path
import shutil

ASSETS = {
	"scenes/cornell_box/meshes/cornell_box.obj": "8569a63a70686d92699df169f16cbb6c75b06d2c9440e62d82b0143d32985bac",
	"scenes/cornell_box/meshes/empty_box.obj": "c5fe2f04929da4bdecde3d1ff49f5bbd7c0140fe21ff5173f117dee6021eecd0",
	"scenes/cow/meshes/cow.obj": "6ce6f4c0f51b8259597ac263fee17216da4751dbc1a6b2ad344a64e2ce6a3260",
	"scenes/lucy/meshes/lucy.obj": "e16300bc5243a3058724e188336fd78995fe6d10a768fbbed57ec7989afea5bd",
}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--source", type=Path, help="Checkout containing the original scenes/*/meshes files"
	)
	args = parser.parse_args()
	root = Path(__file__).resolve().parents[1]
	source = args.source.resolve() if args.source else root
	for relative, expected in ASSETS.items():
		path = source / relative
		if not path.is_file():
			parser.error(
				f"Missing asset: {path}; use --source to restore from a local asset checkout"
			)
		if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
			parser.error(f"Asset checksum mismatch: {path}")
	for relative in ASSETS:
		destination = root / relative
		if source != root:
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(source / relative, destination)
		print(f"Verified {relative}")


if __name__ == "__main__":
	main()
