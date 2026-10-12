#!/usr/bin/env python3
"""Download a free Blendkit original through its official public API.

Accepts an asset base UUID or gallery URL. --output is a directory containing
source.blend, asset-metadata.json and provenance.json. Existing originals are
verified and preserved; interrupted transfers resume only with matching saved
asset/version metadata. No account credentials or signed URLs are recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

API = "https://www.blendkit.com/api/v1/"
USER_AGENT = "Xenon scene downloader/1.0"


class DownloadError(Exception):
	"""An error whose message is safe to display without signed storage URLs."""


def asset_id(value):
	value = value.strip()
	if "://" in value:
		url = urllib.parse.urlsplit(value)
		if url.scheme != "https" or url.hostname not in {"www.blendkit.com", "blendkit.com"}:
			raise argparse.ArgumentTypeError("Use an HTTPS Blendkit gallery URL or asset UUID")
		match = re.fullmatch(r"/asset-gallery-detail/([0-9a-fA-F-]{36})/?", url.path)
		if not match:
			raise argparse.ArgumentTypeError("Expected /asset-gallery-detail/ASSET-UUID/")
		value = match.group(1)
	try:
		return str(uuid.UUID(value))
	except ValueError as error:
		raise argparse.ArgumentTypeError("Invalid asset UUID") from error


def sha256(path):
	digest = hashlib.sha256()
	with path.open("rb") as source:
		while block := source.read(1024 * 1024):
			digest.update(block)
	return digest.hexdigest()


def write_json(path, data):
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
	temporary.replace(path)


def request(url, headers=None):
	return urllib.request.urlopen(
		urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})}),
		timeout=120,
	)


def get_json(url):
	with request(url) as response:
		return json.load(response)


def metadata(base_id):
	query = urllib.parse.urlencode(
		{
			"query": "asset_base_id:" + base_id,
			"page_size": 1,
			"dict_parameters": 1,
			"addon_version": "3.12.2",
		}
	)
	result = get_json(API + "search/?" + query)
	matches = [item for item in result.get("results", []) if item.get("assetBaseId") == base_id]
	if len(matches) != 1:
		raise DownloadError(
			"The public API did not return this asset; check the ID or use Blendkit in Blender"
		)
	item = matches[0]
	if not item.get("isFree") or not item.get("canDownload"):
		raise DownloadError(
			"The official API requires a Blendkit login or entitlement for this asset; use the Blendkit add-on"
		)
	files = [file for file in item.get("files", []) if file.get("fileType") == "blend"]
	if len(files) != 1:
		raise DownloadError("The API did not identify exactly one original blend file")
	endpoint = urllib.parse.urlsplit(files[0].get("downloadUrl", ""))
	if (
		endpoint.scheme != "https"
		or endpoint.hostname != "www.blendkit.com"
		or not endpoint.path.startswith("/api/v1/downloads/")
	):
		raise DownloadError("The asset download endpoint is not the expected official HTTPS API")
	return item, files[0]


def public_metadata(item, file):
	"""Allowlist public fields; never persist any storage/thumbnail URL."""
	keys = (
		"assetBaseId",
		"id",
		"versionNumber",
		"name",
		"description",
		"license",
		"isFree",
		"canDownload",
		"sourceAppName",
		"sourceAppVersion",
		"tags",
		"created",
		"lastUpload",
		"parameters",
		"dictParameters",
	)
	result = {key: item[key] for key in keys if key in item}
	result["author"] = {
		key: (item.get("author") or {}).get(key)
		for key in ("id", "firstName", "lastName", "fullName")
	}
	result["original_file"] = {
		key: file.get(key) for key in ("id", "uuid", "fileType", "fileUploadSize")
	}
	return result


def transfer(url, target, state_path, identity, expected_size):
	partial = target.with_suffix(target.suffix + ".part")
	old_state = json.loads(state_path.read_text()) if state_path.exists() else {}
	if partial.exists() and old_state.get("identity") != identity:
		raise DownloadError(
			"A partial file has missing or different asset/version provenance; preserve it and choose a new output directory"
		)
	offset = partial.stat().st_size if partial.exists() else 0
	if expected_size and offset == expected_size:
		print("Verifying completed partial download", flush=True)
		return partial
	if expected_size and offset > expected_size:
		raise DownloadError(
			"Partial file exceeds the original size; preserve it and choose a new output directory"
		)
	headers = {"Range": f"bytes={offset}-"} if offset else {}
	if offset and old_state.get("etag"):
		headers["If-Range"] = old_state["etag"]
	with request(url, headers) as response:
		append = bool(offset and response.status == 206)
		if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
			raise DownloadError(
				"The server returned an incorrect resume range; the partial file was preserved"
			)
		if response.status not in (200, 206) or (response.status == 206 and not append):
			raise DownloadError("Unexpected file transfer response")
		write_json(state_path, {"identity": identity, "etag": response.headers.get("ETag")})
		count = offset if append else 0
		length = response.headers.get("Content-Length")
		response_total = count + int(length) if length else 0
		print(f"{'Resuming' if append else 'Downloading'} original at {count:,} bytes", flush=True)
		last = time.monotonic()
		with partial.open("ab" if append else "wb") as output:
			while block := response.read(1024 * 1024):
				output.write(block)
				count += len(block)
				if time.monotonic() - last >= 15:
					print(
						f"Downloaded {count:,} / {expected_size or response_total or 'unknown'} bytes",
						flush=True,
					)
					last = time.monotonic()
	if (response_total and count != response_total) or (expected_size and count != expected_size):
		raise DownloadError("Incomplete original file; rerun to resume the saved partial")
	return partial


def download(base_id, output, expected_sha=None):
	output = output.resolve()
	output.mkdir(parents=True, exist_ok=True)
	target, provenance_path = output / "source.blend", output / "provenance.json"
	if target.exists():
		if not provenance_path.exists():
			raise DownloadError(
				"An original already exists without provenance; it was preserved. Choose a new output directory"
			)
		record = json.loads(provenance_path.read_text())
		actual = sha256(target)
		if record.get("asset_base_id") != base_id or record.get("sha256") != actual:
			raise DownloadError(
				"The existing original does not match its asset ID/checksum; it was preserved"
			)
		if expected_sha and expected_sha != actual:
			raise DownloadError("The existing original does not match --sha256; it was preserved")
		print(f"Verified existing original: {target}\nSHA256: {actual}", flush=True)
		return record

	item, file = metadata(base_id)
	state_path = output / ".download-state.json"
	identity = {"asset_base_id": base_id, "asset_version_id": item["id"], "file_id": file["id"]}
	endpoint = file["downloadUrl"]
	separator = "&" if "?" in endpoint else "?"
	link = get_json(
		endpoint + separator + urllib.parse.urlencode({"scene_uuid": str(uuid.uuid4())})
	)
	storage = link.get("filePath")
	if not storage:
		raise DownloadError(
			"The official download API requires login or did not authorize this file; use the Blendkit add-on"
		)
	if urllib.parse.urlsplit(storage).scheme != "https":
		raise DownloadError("The download API did not return an HTTPS file")
	print(
		f"{item.get('name', base_id)} — Blender {item.get('sourceAppVersion', 'unknown')}, license {item.get('license', 'unknown')}",
		flush=True,
	)
	partial = transfer(storage, target, state_path, identity, int(file.get("fileUploadSize") or 0))
	actual = sha256(partial)
	if expected_sha and actual != expected_sha:
		raise DownloadError(
			"Downloaded original does not match --sha256; partial was preserved for inspection"
		)
	author = item.get("author") or {}
	record = {
		**identity,
		"asset_version_number": item.get("versionNumber"),
		"name": item.get("name"),
		"author": (author.get("fullName") or "").strip(),
		"author_id": author.get("id"),
		"license": item.get("license"),
		"is_free": item.get("isFree"),
		"source_application": item.get("sourceAppName"),
		"source_application_version": item.get("sourceAppVersion"),
		"asset_page": "https://www.blendkit.com/asset-gallery-detail/" + base_id + "/",
		"metadata_endpoint": API + "search/?query=asset_base_id:" + base_id,
		"file_type": "blend",
		"sha256": actual,
		"bytes": partial.stat().st_size,
		"retrieved_utc": datetime.now(timezone.utc).isoformat(),
		"source_path": str(target),
		"original_unchanged": True,
		"checksum_verification": "caller-pinned SHA256"
		if expected_sha
		else "local SHA256; Blendkit does not provide an upstream checksum in this API",
		"download_method": "Official public search/download API; no credentials; signed storage URL omitted",
	}
	write_json(output / "asset-metadata.json", public_metadata(item, file))
	write_json(provenance_path, record)
	partial.replace(target)
	state_path.unlink(missing_ok=True)
	print(f"Saved original: {target}\nSHA256: {actual}", flush=True)
	return record


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("asset", type=asset_id, help="Asset base UUID or official gallery URL")
	parser.add_argument(
		"--output", type=Path, required=True, help="Directory for the original and provenance"
	)
	parser.add_argument(
		"--sha256", help="Require a known original checksum for a pinned reproduction"
	)
	args = parser.parse_args(argv)
	if args.sha256 and not re.fullmatch("[0-9a-fA-F]{64}", args.sha256):
		parser.error("--sha256 must contain 64 hexadecimal characters")
	lock = args.output / ".download.lock"
	acquired = False
	try:
		args.output.mkdir(parents=True, exist_ok=True)
		try:
			with lock.open("x") as handle:
				handle.write(
					"Download in progress; remove this lock only after its process has ended.\n"
				)
			acquired = True
		except FileExistsError as error:
			raise DownloadError(
				"Another download owns this output directory; if interrupted, remove .download.lock after confirming its process ended"
			) from error
		download(args.asset, args.output, args.sha256.lower() if args.sha256 else None)
	except urllib.error.HTTPError as error:
		message = (
			"Login or entitlement required; use the official Blendkit add-on"
			if error.code in (401, 403)
			else "Official download service request failed"
		)
		print(
			f"{message} (HTTP {error.code}); original/partial files were preserved", file=sys.stderr
		)
		return 1
	except DownloadError as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1
	except (
		OSError,
		ValueError,
		KeyError,
		urllib.error.URLError,
		http.client.HTTPException,
	) as error:
		# Exception messages may include signed request URLs; only show their type.
		print(
			f"Download failed ({type(error).__name__}); original/partial files were preserved",
			file=sys.stderr,
		)
		return 1
	finally:
		if acquired:
			lock.unlink(missing_ok=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
