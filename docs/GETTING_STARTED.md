# Install and run

Xenon is developed and tested on x86-64 Linux and WSL. The current build uses AVX2/FMA and native CPU optimization, so build it on the machine where it will run. It needs a C++20 compiler, CMake 3.20+, Python 3, GLFW and OpenGL development libraries, even for headless output.

On Ubuntu:

```sh
sudo apt install build-essential cmake pkg-config libglfw3-dev libgl1-mesa-dev python3 git
git clone https://github.com/Tethane/Xenon.git
cd Xenon
./xenon.sh --scene prim_types --backend cpu --samples 64
```

This scene uses analytic primitives and works without external meshes. The window progressively updates as samples accumulate. Add `--no-display` for a terminal-only render.

## The launcher

`./xenon.sh` is the rendering entry point. It configures and incrementally builds the executable, resolves scene names, and forwards rendering options. You can still call a built `xenon` binary directly. Help and scene listing do not trigger a build.

The default directories are `build/release` and `build/debug`. Set `XENON_BUILD_DIR` to use another CMake build (relative paths are resolved from the repository root). `XENON_JOBS` controls build parallelism; `CMAKE_BUILD_PARALLEL_LEVEL` is also respected. The launcher builds only the renderer target.

```sh
./xenon.sh --help
./xenon.sh --list-scenes
./xenon.sh --scene cornell_box --samples 256 --backend cpu
./xenon.sh --scene lucy --samples 512 --backend optix
./xenon.sh --scene sponza --samples 1024 --backend optix
./xenon.sh --scene classroom --samples 1024 --backend optix
./xenon.sh --scene /path/to/custom.xenon --samples 512 --backend cuda
```

Original names such as `cornell_box`, `lucy`, `cow`, `prim_types` and `sky_dome` remain supported. `sponza` and `classroom` resolve through the [benchmark manifest](../scenes/benchmarks/suite.json), as do its full scene IDs. They require the generated exports described below. Sponza uses the manifest's +4 EV display exposure; an explicit `--exposure` overrides it and never changes linear output.

## GPU builds

Install a compatible NVIDIA driver and CUDA Toolkit. Make `nvcc` available when requesting a GPU backend, or set `CUDACXX` to its full path. A fresh CPU request configures without CUDA; a later explicit GPU request enables it. An explicit CMake build gives access to all configuration options:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DXENON_ENABLE_CUDA=ON -DXENON_ENABLE_OPTIX=ON
cmake --build build/release --target xenon --parallel 4
./xenon.sh --scene prim_types --backend optix --samples 64
```

CMake fetches pinned OptiX headers. An existing SDK can be selected with `-DOPTIX_INCLUDE_DIR=/path/to/OptiX/include`. Requested unavailable backends fail explicitly. The launcher reconfigures when a requested GPU backend needs to be enabled. For manual CMake builds, configure again after installing CUDA and inspect the backend messages.

For a CPU-only configuration:

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
  -DXENON_ENABLE_CUDA=OFF -DXENON_ENABLE_OPTIX=OFF
cmake --build build/cpu --target xenon --parallel 4
XENON_BUILD_DIR=build/cpu ./xenon.sh --scene prim_types --backend cpu
```

[OptiX setup](OPTIX.md) documents runtime requirements and the isolated WSL workaround used for the published measurements. For an OptiX render on WSL, the launcher uses the already-installed `build/optix-runtime` libraries when present. It does not download them. The workaround is specific to the recorded driver combination; it is not a general driver installer.

## Scenes and assets

**Existing checkout:** the downloaded/exported scenes from qualification and the original meshes can be used directly. `--list-scenes` shows scene-file availability; the renderer validates referenced assets when loading.

**Fresh clone:** `prim_types`, `sky_dome`, `outdoor_sun` and `instancing_grid` need no OBJ download. Original Cornell, Lucy and cow meshes are excluded from Git. Restore them from a checkout that contains them:

```sh
python3 scripts/setup_assets.py --source /path/to/original/Xenon
# With assets already in place, verify their checksums:
python3 scripts/setup_assets.py
```

The old scene-generator scripts are not an installer for those meshes.

For Sponza, Classroom and the studios:

```sh
python3 scripts/scene_assets.py blender classroom sponza studio_hdri
```

Then follow [the Blender source/export commands](BLENDER_IMPORT.md) and [final-suite variant commands](BLENDER_IMPORT.md#reproduce-the-final-five-scene-suite). Downloads live in `build/assets/`, Blender in `build/tools/`, and exported scenes in `build/fixtures/`. Downloading an archive alone does not create its Xenon scene. Sponza's archive is approximately 4 GB; extraction and exports need additional disk space.

Your own `.blend`, glTF/GLB or OBJ scene uses the same [import workflow](BLENDER_IMPORT.md). [Blendkit and grass preparation](BLENDER_IMPORT.md#blendkit-scenes-and-grass) covers scatter density and hair conversion.

## Output and controls

```sh
./xenon.sh --scene lucy --backend optix --no-display \
  --width 800 --height 600 --samples 1024 --seed 42 \
  --min-bounces 3 --max-bounces 8 --exposure 0 \
  --output-dir renders/lucy --output beauty.png \
  --linear-output radiance.pfm --metrics metrics.json
```

PNG and preview share exposure, Reinhard tone mapping and sRGB conversion. PFM stores unclamped scene-linear float32 RGB; JSON records settings, hardware and stage timings. Relative output filenames resolve under `--output-dir` (default `renders/`); absolute paths are used directly. Scene-relative asset paths are resolved by the loader.

`--bounces` and `--outdir` remain aliases for `--max-bounces` and `--output-dir`. `--build` reconfigures; `--debug` selects a Debug build. Closing the preview before completion returns exit status 2, a failure returns 1, and a completed render returns 0.

Generated output is ignored by Git. Published images belong in [docs/renders/](renders/README.md), alongside their settings and attribution.

## Development checks

Use targeted tests while making local changes. Launcher checks run without a compiler or GPU:

```sh
python3 tests/launcher_test.py
```

The complete recorded backend validation is in [QUALIFICATION.md](QUALIFICATION.md). To configure and run backend tests:

```sh
cmake -S . -B build/release -DXENON_ENABLE_CUDA=ON \
  -DXENON_ENABLE_OPTIX=ON -DXENON_TEST_CUDA=ON
cmake --build build/release --parallel 4
ctest --test-dir build/release --output-on-failure
```

GPU execution checks require a working NVIDIA device. Actual OpenGL checks are opt-in through `XENON_TEST_DISPLAY=ON`; Debug CPU builds enable ASan/UBSan. Python comparison tools need NumPy, and comparison figures additionally need Matplotlib.
