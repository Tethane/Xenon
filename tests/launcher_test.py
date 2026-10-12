"""Focused launcher checks using a temporary repository and mocked CMake."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


with tempfile.TemporaryDirectory(prefix="xenon-launcher-") as directory:
	root = Path(directory)
	shutil.copy2(Path(__file__).resolve().parents[1] / "xenon.sh", root / "xenon.sh")
	for name in ("cornell_box", "prim_types", "lucy"):
		scene = root / "scenes" / name / f"{name}.xenon"
		scene.parent.mkdir(parents=True)
		scene.write_text("config 32 24 1\n")
	(root / "scenes/cornell_box/cornell_empty.xenon").write_text("config 32 24 1\n")
	manifest = root / "scenes/benchmarks/suite.json"
	manifest.parent.mkdir()
	entries = []
	for name in ("sponza-full", "classroom-instances"):
		scene = root / "build/fixtures" / name / "scene.xenon"
		scene.parent.mkdir(parents=True)
		scene.write_text("config 800 600 128\n")
		entries.append(
			dict(
				id=name,
				scene=str(scene.relative_to(root)),
				width=800,
				height=600,
				exposure=4 if name == "sponza-full" else 0,
			)
		)
	manifest.write_text(
		json.dumps(dict(defaults=dict(min_bounces=3, max_bounces=8, seed=42), scenes=entries))
	)
	tools = root / "tools"
	tools.mkdir()
	log = root / "cmake.jsonl"
	cmake = tools / "cmake"
	cmake.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['MOCK_LOG'],'a') as f:f.write(json.dumps(a)+'\\n')
if '--build' in a:
    sys.exit(int(os.environ.get('MOCK_BUILD_EXIT','0')))
if os.environ.get('MOCK_CONFIG_FAIL'):sys.exit(7)
b=Path(a[a.index('-B')+1]);b.mkdir(parents=True,exist_ok=True)
cache=b/'CMakeCache.txt';v={}
if cache.exists():
    v=dict(line.split('=',1) for line in cache.read_text().splitlines())
for arg in a:
    if arg.startswith('-D'):
        key,value=arg[2:].split('=',1);v[key+':STRING']=value
if v.get('XENON_ENABLE_CUDA:STRING')=='ON':
    v['CMAKE_CUDA_COMPILER:FILEPATH']='NOTFOUND' if os.environ.get('MOCK_NO_CUDA') else '/mock/nvcc'
cache.write_text(''.join(k+'='+val+'\\n' for k,val in v.items()))
binary=b/'xenon'
binary.write_text('#!/usr/bin/env python3\\nimport json,os,sys\\nprint("ARGS="+json.dumps(sys.argv[1:]))\\nprint("LOADER="+os.environ.get("LD_LIBRARY_PATH",""))\\n')
binary.chmod(0o755)
""")
	cmake.chmod(0o755)
	for name, output in (("uname", "mock-microsoft-WSL2"), ("getconf", "64")):
		path = tools / name
		path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
		path.chmod(0o755)
	env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"], "MOCK_LOG": str(log)}
	for key in ("XENON_BUILD_DIR", "XENON_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
		env.pop(key, None)

	def run(*args, success=True, extra=None):
		result = subprocess.run(
			[str(root / "xenon.sh"), *args],
			cwd="/tmp",
			env={**env, **(extra or {})},
			capture_output=True,
			text=True,
		)
		assert (result.returncode == 0) == success, result.stdout + result.stderr
		return result

	def calls():
		return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

	def arguments(result):
		return json.loads(
			next(line[5:] for line in result.stdout.splitlines() if line.startswith("ARGS="))
		)

	def last(args, key):
		return args[max(i for i, x in enumerate(args) if x == key) + 1]

	assert "Usage:" in run("--help").stdout
	listing = run("--list-scenes").stdout
	assert "sponza / sponza-full" in listing and "classroom / classroom-instances" in listing
	assert "cornell_empty" in listing
	for args in [("--backend", "bogus"), ("--samples",), ("--scene", "missing"), ("--unknown",)]:
		run(*args, success=False)
	assert not calls(), "Help, list and invalid arguments must not build"
	args = arguments(run("--scene", "prim_types", "--backend", "cpu"))
	assert last(args, "--scene") == str(root / "scenes/prim_types/prim_types.xenon")
	assert "-DXENON_ENABLE_CUDA=OFF" in calls()[0] and "-DXENON_ENABLE_OPTIX=OFF" in calls()[0]
	assert calls()[-1][-2:] == ["--parallel", "4"] and calls()[-1][2:4] == ["--target", "xenon"]
	args = arguments(run("--scene", "cornell_empty", extra={"CMAKE_BUILD_PARALLEL_LEVEL": "3"}))
	assert last(args, "--scene") == str(root / "scenes/cornell_box/cornell_empty.xenon")
	assert calls()[-1][-2:] == ["--parallel", "3"]
	runtime = root / "build/optix-runtime"
	runtime.mkdir()
	(runtime / "libnvoptix.so.1").touch()
	result = run(
		"--scene",
		"sponza",
		"--backend",
		"optix",
		"--width",
		"17",
		"--exposure",
		"-2",
		"--bounces",
		"6",
		"--outdir",
		"output space",
	)
	args = arguments(result)
	assert last(args, "--scene") == str(root / entries[0]["scene"])
	assert last(args, "--width") == "17" and last(args, "--height") == "600"
	assert last(args, "--exposure") == "-2" and args[args.index("--exposure") + 1] == "4"
	assert last(args, "--max-bounces") == "6" and last(args, "--seed") == "42"
	assert (
		last(args, "--output-dir") == "output space"
		and last(args, "--output") == "sponza_output.png"
	)
	assert "-DXENON_ENABLE_CUDA=ON" in calls()[-2] and "-DXENON_ENABLE_OPTIX=ON" in calls()[-2]
	assert f"LOADER={runtime}:/usr/lib/wsl/lib" in result.stdout
	before = len(calls())
	run("--scene", "lucy", "--backend", "cpu")
	assert len(calls()) == before + 1, "CPU request should preserve an existing GPU configuration"
	args = arguments(run("--scene", "classroom", "--backend", "optix"))
	assert (
		last(args, "--scene") == str(root / entries[1]["scene"]) and last(args, "--exposure") == "0"
	)
	custom = root / "custom scene.xenon"
	custom.write_text("config 1 1 1\n")
	args = arguments(
		run(
			"--scene",
			str(custom),
			"--debug",
			"--build",
			extra={"XENON_BUILD_DIR": "build/custom space", "XENON_JOBS": "2"},
		)
	)
	assert last(args, "--scene") == str(custom) and calls()[-1][-2:] == ["--parallel", "2"]
	assert (
		"-DCMAKE_BUILD_TYPE=Debug" in calls()[-2]
		and str(root / "build/custom space") in calls()[-1]
	)
	run("--scene", "prim_types", extra={"XENON_JOBS": "0"}, success=False)
	run("--scene", "prim_types", extra={"MOCK_BUILD_EXIT": "9"}, success=False)
	failure = run(
		"--scene",
		"prim_types",
		"--backend",
		"cuda",
		extra={"XENON_BUILD_DIR": "build/no-cuda", "MOCK_NO_CUDA": "1"},
		success=False,
	)
	assert "no CPU fallback" in failure.stderr and "--build" not in calls()[-1]
	run("--scene", "prim_types", "--backend", "cuda", extra={"XENON_BUILD_DIR": "build/no-cuda"})
	assert "-UCMAKE_CUDA_COMPILER" in calls()[-2], (
		"Retry must clear a stale missing-compiler cache entry"
	)
	(root / entries[0]["scene"]).unlink()
	before = len(calls())
	assert "not prepared" in run("--scene", "sponza", success=False).stderr
	assert len(calls()) == before, "Missing benchmark must fail before building"
print(
	"Launcher help, aliases/default overrides, legacy flags, build selection and failure checks passed."
)
