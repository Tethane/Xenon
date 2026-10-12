#!/usr/bin/env python3
"""Download pinned official Blender/Classroom/Sponza assets outside version control.

The default cache is build/assets. Large downloads are streamed and SHA-256
verified before extraction. Run --list to inspect provenance without downloads.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

ASSETS = {
	"blender": {
		"filename": "blender-4.5.3-linux-x64.tar.xz",
		"url": "https://mirror.blender.org/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz",
		"sha256": "975c58fcb244273838534bba771e64ad87739216b0f9b39a888531a49a72d845",
		"bytes": 377397328,
		"author": "Blender Foundation and contributors",
		"license": "GPL-3.0-or-later",
		"provenance": "https://www.blender.org/download/",
	},
	"classroom": {
		"filename": "classroom.zip",
		"url": "https://mirror.blender.org/demo/test/classroom.zip",
		"sha256": "0f0ecc58d45f12b2f4272a53e2d8e69135518e16e956b118f10c20d8dcfdf5e6",
		"bytes": 70279690,
		"author": "Christophe Seux",
		"license": "CC0",
		"provenance": "https://www.blender.org/download/demo-files/",
	},
	"studio_hdri": {
		"filename": "studio_small_09_1k.hdr",
		"url": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr",
		"sha256": "e7cfda5f4e98e623db12b8bfd0184e048488e4855d9c83e2751fb44a32e80c45",
		"author": "Poly Haven",
		"license": "CC0",
		"provenance": "https://polyhaven.com/a/studio_small_09",
		"extract": False,
	},
	"sponza": {
		"filename": "main_sponza.zip",
		"url": "https://cdrdv2.intel.com/v1/dl/getContent/830833",
		"sha256": "b8bb853884ab1566b3beb35666bd09882a4e0dc16661e4684e103792cf0229b9",
		"bytes": 3987608266,
		"author": "Intel GPU Research; Sponza 2022 base scene",
		"license": "CC-BY-4.0 text plus introductory usage notice; retain credits_license.txt",
		"provenance": "https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html",
	},
}


def digest(path):
	result = hashlib.sha256()
	with path.open("rb") as f:
		while block := f.read(1024 * 1024):
			result.update(block)
	return result.hexdigest()


def download(spec, downloads):
	target = downloads / spec["filename"]
	expected = spec["sha256"]
	if expected == "PENDING_DOWNLOAD":
		raise RuntimeError("Sponza checksum has not yet been pinned; refuse unverified download")
	if target.exists() and digest(target) == expected:
		print(f"Verified cached {target}", flush=True)
		return target
	partial = target.with_suffix(target.suffix + ".part")
	print(f"Downloading {spec['url']} to {partial}", flush=True)
	request = urllib.request.Request(
		spec["url"], headers={"User-Agent": "Xenon scene asset setup/1.0"}
	)
	with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as output:
		shutil.copyfileobj(source, output, length=1024 * 1024)
	actual = digest(partial)
	if actual != expected:
		raise RuntimeError(f"SHA-256 mismatch for {partial}: expected {expected}, got {actual}")
	partial.replace(target)
	return target


def safe_extract(archive, destination):
	destination.mkdir(parents=True, exist_ok=True)
	root = destination.resolve()
	if zipfile.is_zipfile(archive):
		with zipfile.ZipFile(archive) as z:
			for member in z.infolist():
				path = (root / member.filename).resolve()
				if path != root and root not in path.parents:
					raise RuntimeError(f"Unsafe archive path: {member.filename}")
				if member.is_dir():
					path.mkdir(parents=True, exist_ok=True)
				else:
					path.parent.mkdir(parents=True, exist_ok=True)
					with z.open(member) as source, path.open("wb") as output:
						shutil.copyfileobj(source, output)
	else:
		with tarfile.open(archive) as tar:
			tar.extractall(root, filter="data")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("assets", nargs="*", default=[])
	parser.add_argument("--directory", type=Path, default=Path("build/assets"))
	parser.add_argument("--tools-directory", type=Path, default=Path("build/tools"))
	parser.add_argument("--list", action="store_true")
	args = parser.parse_args()
	if args.list:
		print(json.dumps(ASSETS, indent=2))
		return
	downloads = args.directory / "downloads"
	downloads.mkdir(parents=True, exist_ok=True)
	results = []
	for name in args.assets or list(ASSETS):
		if name not in ASSETS:
			parser.error(f"Unknown asset: {name}; choose from {list(ASSETS)}")
		spec = ASSETS[name]
		archive = download(spec, downloads)
		destination = args.tools_directory if name == "blender" else args.directory / name
		marker = destination / f".{name}-{spec['sha256']}.extracted"
		if not marker.exists():
			print(f"Extracting {archive} into {destination}", flush=True)
			if spec.get("extract", True):
				safe_extract(archive, destination)
			else:
				destination.mkdir(parents=True, exist_ok=True)
				shutil.copyfile(archive, destination / spec["filename"])
			marker.write_text(spec["sha256"] + "\n")
		record = {
			**spec,
			"asset": name,
			"archive": str(archive.resolve()),
			"directory": str(destination.resolve()),
		}
		results.append(record)
		(destination / f"{name}-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
