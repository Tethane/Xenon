# OptiX and queue scheduling experiment

Measured on 2026-10-12: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, Windows driver
592.00 through WSL, local Linux OptiX runtime 590.48.01, OptiX 9.0 headers,
Release CUDA architecture 89. The scene is the shared Blender material studio,
10,786 triangles, textured surfaces, glass and emitting triangles. Resolution
512×384, 128 samples, seed 42, minimum/maximum bounces 3/8, no preview.

All variants below use **the same executable and material implementation**.
Each variant has one discarded warmup and three measured runs, interleaved in
round-robin order. The frozen executable is
`build/optimization/controlled-xenon`; its SHA-256 and individual timings are in
`build/optimization/controlled/results.json`. Further feature development after
this experiment may change absolute timings.

| Backend and scheduling | Median render (s) | Range (s) | Median startup through image output (s) |
|---|---:|---:|---:|
| Software CUDA, host counters | 2.844 | 2.757–2.878 | 3.516 |
| OptiX, host counters and parameter uploads | 1.494 | 1.443–1.581 | 2.536 |
| OptiX, host counters and cached parameters | 1.536 | 1.475–1.585 | 2.485 |
| Software CUDA, device counters | 1.788 | 1.785–1.812 | 2.499 |
| OptiX, device counters and cached parameters | 0.484 | 0.455–0.535 | 1.528 |

With device counters on both backends, OptiX gives a **3.70× rendering speedup**
and a **1.63× speedup through image output** on this scene. Device counters improve
software CUDA by 1.59× and OptiX by **3.18×**, holding parameter caching enabled
for both OptiX configurations.
Parameter caching alone has no demonstrated timing gain in these repetitions.
These are scene-specific results, not a universal performance claim.

The reported `seconds.total` covers startup through image output, including scene
loading and acceleration construction. It stops before writing the metrics JSON
and tearing down the renderer. The general benchmark runner separately measures
`wall_seconds` across the complete subprocess; these timings have different
boundaries and should retain their labels.

For each backend, scheduling variants produce identical PFM bytes. OptiX and
software CUDA differ by relative linear RMSE **0.008676%** and relative mean
luminance **0.000120%** with the same seeds. Hardware and software triangle
intersection rounding can change a few paths; this comparison does not prove
agreement with Blender's different material model.

## Profile evidence

Nsight Systems 2026.1.1 profiles used the same executable at 32 samples. The
render interval spans the first ray-generation kernel through the final
accumulation kernel. GPU busy time is the union of kernels, transfers and
memsets. Uncovered intervals are measured directly instead of inferred from CPU
API waiting time.

| OptiX configuration | Render interval | GPU busy | Uncovered intervals | Busy fraction |
|---|---:|---:|---:|---:|
| Host counters and uploads | 478.06 ms | 104.60 ms | 373.45 ms | 21.9% |
| Device counters and cache | 143.28 ms | 115.27 ms | 28.02 ms | 80.4% |

Synchronous `cudaMemcpy` calls within the interval fall from 2,559 to three
initial parameter uploads. Queue advancement instead uses 256 asynchronous
device-to-device copies. Fixed-capacity launches execute some inactive threads,
increasing GPU work by about 10%, while uncovered intervals fall by 92.5%.
One synchronization at each sample boundary preserves cancellation and timing.

`cudaDeviceSynchronize` waiting actually rises from about 3 ms to 79 ms while
the render interval becomes 3.34× shorter. Earlier synchronization points had
absorbed that GPU execution time. Its CPU duration alone is therefore not a
measure of synchronization overhead.

## Reproduction

Set up the optional WSL runtime as described in [OPTIX.md](OPTIX.md), then run:

```sh
python3 scripts/benchmark_gpu_scheduling.py build/fixtures/material_studio/scene.xenon \
  --binary build/optix/xenon --samples 128 --repeats 3

nsys profile --trace=cuda --sample=none --cpuctxsw=none -o build/device-profile \
  build/optix/xenon --scene build/fixtures/material_studio/scene.xenon \
  --backend optix --width 512 --height 384 --samples 32 --max-bounces 8 \
  --seed 42 --no-display
nsys export --type sqlite -o build/device-profile.sqlite build/device-profile.nsys-rep
python3 scripts/analyze_gpu_profile.py build/device-profile.sqlite
```

Repeat the profile with `XENON_GPU_HOST_COUNTERS=1 XENON_OPTIX_UPLOAD_PARAMS=1`
to obtain the control. The retained reports are
`build/optimization/{host,device}-profile.nsys-rep` and their JSON summaries.

The optimized binaries passed CUDA and OptiX linear rendering and texture tests,
including opacity visibility, normals, bump maps and nonuniform transforms.
Separate OptiX geometry tests cover analytic primitives, infinite planes,
mirrored instances, finite shadow intervals and empty scenes. These tests remain
part of CTest; benchmark data alone does not replace them.
