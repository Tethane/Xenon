#!/usr/bin/env python3
"""Downloader contract tests; no network calls or Blendkit account required."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

SPEC = importlib.util.spec_from_file_location(
	"download_blendkit", Path(__file__).resolve().parents[1] / "scripts/download_blendkit.py"
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)
ID = "fea95d8b-bb8d-4bdd-bd53-e1a6252fa3b7"
VERSION = "ca34ad8d-9646-4f47-ab3b-5f390e6ad775"
SECRET_URL = "https://storage.example/original.blend?signature=DO_NOT_LOG"


class Response(io.BytesIO):
	def __init__(self, data, status=200, headers=None):
		super().__init__(data)
		self.status = status
		self.headers = {"Content-Length": str(len(data)), **(headers or {})}


class DownloaderTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.output = Path(self.directory.name)
		self.target = self.output / "source.blend"
		self.state = self.output / ".download-state.json"
		self.identity = {"asset_base_id": ID, "asset_version_id": VERSION, "file_id": 1}

	def test_id_and_official_gallery_url(self):
		self.assertEqual(downloader.asset_id(ID), ID)
		self.assertEqual(
			downloader.asset_id("https://www.blendkit.com/asset-gallery-detail/" + ID + "/"), ID
		)
		for value in ("https://evil.example/asset-gallery-detail/" + ID, "not-an-id"):
			with self.assertRaises(argparse.ArgumentTypeError):
				downloader.asset_id(value)

	def test_existing_original_verified_without_network(self):
		self.target.write_bytes(b"original")
		downloader.write_json(
			self.output / "provenance.json",
			{**self.identity, "sha256": downloader.sha256(self.target)},
		)
		with patch.object(
			downloader, "get_json", side_effect=AssertionError("No request expected")
		):
			downloader.download(ID, self.output)

	def test_output_file_is_preserved_without_traceback_or_network(self):
		existing = self.output / "existing.txt"
		existing.write_text("Keep this file")
		log = io.StringIO()
		with (
			patch.object(
				downloader, "download", side_effect=AssertionError("No download expected")
			),
			contextlib.redirect_stderr(log),
		):
			self.assertEqual(downloader.main([ID, "--output", str(existing)]), 1)
		self.assertEqual(existing.read_text(), "Keep this file")
		self.assertIn("Download failed (FileExistsError)", log.getvalue())
		self.assertNotIn("Traceback", log.getvalue())

	def test_unproven_existing_original_preserved(self):
		self.target.write_bytes(b"precious existing original")
		with self.assertRaises(downloader.DownloadError):
			downloader.download(ID, self.output)
		self.assertEqual(self.target.read_bytes(), b"precious existing original")

	def test_original_checksum_mismatch_preserved(self):
		self.target.write_bytes(b"changed")
		downloader.write_json(
			self.output / "provenance.json", {**self.identity, "sha256": "0" * 64}
		)
		with self.assertRaises(downloader.DownloadError):
			downloader.download(ID, self.output)
		self.assertEqual(self.target.read_bytes(), b"changed")

	def test_matching_partial_resumes(self):
		self.target.with_suffix(".blend.part").write_bytes(b"abc")
		downloader.write_json(self.state, {"identity": self.identity, "etag": "same-file"})
		response = Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})
		with patch.object(downloader, "request", return_value=response) as request:
			partial = downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6)
		self.assertEqual(partial.read_bytes(), b"abcdef")
		self.assertEqual(request.call_args.args[1], {"Range": "bytes=3-", "If-Range": "same-file"})
		self.assertNotIn("DO_NOT_LOG", self.state.read_text())

	def test_wrong_range_preserves_partial(self):
		partial = self.target.with_suffix(".blend.part")
		partial.write_bytes(b"abc")
		downloader.write_json(self.state, {"identity": self.identity})
		with patch.object(
			downloader,
			"request",
			return_value=Response(b"bad", 206, {"Content-Range": "bytes 1-3/6"}),
		):
			with self.assertRaises(downloader.DownloadError):
				downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6)
		self.assertEqual(partial.read_bytes(), b"abc")

	def test_completed_partial_is_verified_without_transfer(self):
		partial = self.target.with_suffix(".blend.part")
		partial.write_bytes(b"abcdef")
		downloader.write_json(self.state, {"identity": self.identity})
		with patch.object(
			downloader, "request", side_effect=AssertionError("No transfer expected")
		):
			self.assertEqual(
				downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6), partial
			)

	def test_unproven_partial_preserved(self):
		partial = self.target.with_suffix(".blend.part")
		partial.write_bytes(b"unknown")
		with patch.object(downloader, "request", side_effect=AssertionError("No request expected")):
			with self.assertRaises(downloader.DownloadError):
				downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6)
		self.assertEqual(partial.read_bytes(), b"unknown")

	def test_server_ignoring_range_restarts_same_asset(self):
		self.target.with_suffix(".blend.part").write_bytes(b"abc")
		downloader.write_json(self.state, {"identity": self.identity})
		with patch.object(downloader, "request", return_value=Response(b"abcdef")):
			partial = downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6)
		self.assertEqual(partial.read_bytes(), b"abcdef")

	def test_incomplete_download_does_not_promote(self):
		with patch.object(downloader, "request", return_value=Response(b"abc")):
			with self.assertRaises(downloader.DownloadError):
				downloader.transfer(SECRET_URL, self.target, self.state, self.identity, 6)
		self.assertFalse(self.target.exists())

	def test_denied_asset_reports_login(self):
		item = {"assetBaseId": ID, "isFree": True, "canDownload": False}
		with patch.object(downloader, "get_json", return_value={"results": [item]}):
			with self.assertRaisesRegex(downloader.DownloadError, "login"):
				downloader.metadata(ID)

	def test_complete_download_records_no_signed_url(self):
		item = {
			"assetBaseId": ID,
			"id": VERSION,
			"name": "Fixture",
			"sourceAppVersion": "5.2.0",
			"license": "royalty_free",
			"isFree": True,
			"author": {"fullName": "Artist"},
		}
		file = {
			"id": 1,
			"fileType": "blend",
			"fileUploadSize": 6,
			"downloadUrl": downloader.API + "downloads/fixture/",
		}
		log = io.StringIO()
		with (
			patch.object(downloader, "metadata", return_value=(item, file)),
			patch.object(downloader, "get_json", return_value={"filePath": SECRET_URL}),
			patch.object(downloader, "request", return_value=Response(b"abcdef")),
			contextlib.redirect_stdout(log),
		):
			record = downloader.download(ID, self.output)
		self.assertEqual(self.target.read_bytes(), b"abcdef")
		self.assertEqual(record["bytes"], 6)
		self.assertNotIn("DO_NOT_LOG", log.getvalue())
		for path in self.output.glob("*.json"):
			self.assertNotIn("DO_NOT_LOG", path.read_text())

	def test_http_error_does_not_log_signed_url(self):
		log = io.StringIO()
		error = urllib.error.HTTPError(SECRET_URL, 403, "DO_NOT_LOG", {}, None)
		with (
			patch.object(downloader, "download", side_effect=error),
			contextlib.redirect_stderr(log),
		):
			self.assertEqual(downloader.main([ID, "--output", str(self.output)]), 1)
		self.assertNotIn("DO_NOT_LOG", log.getvalue())
		self.assertIn("Login", log.getvalue())
		self.assertFalse((self.output / ".download.lock").exists())


if __name__ == "__main__":
	unittest.main()
