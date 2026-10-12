# OptiX traversal

The OptiX backend runs Xenon's CUDA wavefront integrator with OptiX closest-hit
and shadow queries. Material evaluation, random streams, lights, BSDFs,
accumulation and display transforms are shared with software CUDA.

The build fetches NVIDIA's official minimal headers at
[`optix-dev` v9.0.0](https://github.com/NVIDIA/optix-dev/tree/fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd).
Pass `-DOPTIX_INCLUDE_DIR=/path/to/OptiX/include` for an existing installation,
or `-DXENON_ENABLE_OPTIX=OFF` to retain only software CUDA. Builds without CUDA
default to disabling OptiX. The PTX programs are embedded in the executable.

```sh
cmake -S . -B build/optix -DCMAKE_BUILD_TYPE=Release -DXENON_TEST_CUDA=ON
cmake --build build/optix -j
build/optix/xenon --backend optix --scene scenes/cornell_box/cornell_box.xenon --no-display
ctest --test-dir build/optix -R 'optix|cuda_texture' --output-on-failure
```

Each mesh has one compacted triangle GAS, shared by its instances. A top-level
IAS retains instance transforms and stable Xenon hit/material IDs. Spheres,
boxes, disks and quads use custom OptiX intersections with the same equations
as CPU/CUDA. Infinite planes are handled through an explicit plane list because
their support cannot be represented by finite AABBs. Opacity cutouts affect both
closest-hit and shadow rays. Rays retain Xenon's `[tmin,tmax)` interval.

`--backend optix` fails when OptiX is unavailable. It never selects software
CUDA as a fallback. Measurements report the OptiX acceleration build separately
from scene upload; the software BVHs are currently still built and uploaded as
part of the shared scene representation, and those costs remain in totals.

Queue counters stay on the device by default, with one synchronization per
sample for timing/cancellation and optional preview publication. OptiX launch
parameters are cached across the two ray queues and shadow pass. Diagnostic
controls `XENON_GPU_HOST_COUNTERS=1` and `XENON_OPTIX_UPLOAD_PARAMS=1` restore
host counter reads and per-launch parameter uploads independently. They do not
change material or transport equations. Compare these controls with one binary:

```sh
python3 scripts/benchmark_gpu_scheduling.py build/fixtures/material_studio/scene.xenon \
  --binary build/optix/xenon --repeats 3 --samples 128
```

This writes linear images, timing spread, control settings, executable and scene
hashes, and image errors. Rendering speedup and startup-through-image-output
speedup can differ; scene loading and acceleration construction remain in the
latter. The renderer's `seconds.total` stops after image output, before metrics
output and teardown. The general benchmark runner also records the complete
subprocess as `wall_seconds`, and labels the corresponding speedups
`through_image_output` and `process_wall`.

GPU memory is an approximate indicator: the console's scene-upload MB currently
omits HDR environment pixels and their sampling CDF. The JSON free-memory delta
includes those allocations, along with context and allocator effects, so it is
not an exact allocation count.

## This machine's WSL runtime

The Windows driver exposes a `libnvoptix.so.1` shim without the OptiX entry
point. NVIDIA describes an
[experimental WSL procedure](https://forums.developer.nvidia.com/t/running-optix-on-wsl-2026-version/382414)
using matching Linux userspace runtime libraries. Xenon's helper extracts
590.48.01 OptiX, RTCore and GPU compiler libraries into the ignored build
directory. It does not install a driver or replace system libraries:

```sh
bash scripts/setup_optix_wsl.sh
export LD_LIBRARY_PATH="$PWD/build/optix-runtime:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir build/optix -R optix --output-on-failure
```

This particular library set was selected for the installed 590-branch WSL
driver (Windows reports 592.00). It is not a general installer for other driver
branches. NVIDIA does not officially support this WSL combination; native Linux
or Windows drivers with OptiX support do not require the extraction helper.
