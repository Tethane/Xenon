# Xenon

A physically based path tracer written from scratch in C++ with an optional CUDA GPU backend. Xenon implements unbiased Monte Carlo light transport with next event estimation (NEE) and multiple importance sampling (MIS) to produce high-quality rendered images of 3D scenes.

Both the CPU and GPU renderers share the same wavefront pipeline architecture and produce equivalent output.

## Example Renders

<p align="center">
  <img src="renders/cornell_box_output.png" width="400" alt="Cornell Box — area lighting with color bleeding" />
  <img src="renders/cow_output.png" width="400" alt="Metallic cow — GGX microfacet reflections" />
</p>
<p align="center">
  <img src="renders/lucy_output.png" width="400" alt="Stanford Lucy — glass with full transmission and caustics" />
  <img src="renders/instancing_grid_output.png" width="400" alt="Instancing grid — multiple material types with environment lighting" />
</p>
<p align="center">
  <img src="renders/prim_types_output.png" width="400" alt="Analytic primitives — sphere, box, disk, quad with glass" />
  <img src="renders/combined_outdoor_output.png" width="400" alt="Outdoor scene — directional sunlight with gradient sky" />
</p>

All images above are rendered directly by Xenon at 256 samples per pixel.

## Features

### Rendering
- Unbiased path tracing with configurable bounce depth (Russian roulette termination)
- Next event estimation (NEE) with direct light sampling
- Multiple importance sampling (MIS) using the balance heuristic
- Progressive accumulation with live preview via GLFW/OpenGL

### Materials
- Principled BSDF supporting diffuse, conductor, dielectric, and subsurface lobes
- GGX microfacet model for rough specular reflections and transmissions
- Fresnel-weighted lobe selection for glass, metal, and plastic surfaces
- File-based material definitions (`.mat` format) with hot-reload support

### Geometry
- Triangle meshes loaded from Wavefront OBJ files
- Analytic primitives: sphere, box, plane, disk, quad
- Two-level acceleration structure (TLAS/BLAS) with binned SAH BVH construction
- Object instancing with per-instance affine transforms

### Lighting
- Area lights (emissive triangles)
- Directional lights (sun)
- Gradient sky environment model

### GPU Backend (CUDA)
- Full wavefront pipeline reimplemented in CUDA kernels
- Scene data flattened and uploaded to device memory once at startup
- BVH traversal, material evaluation, and light sampling all run on the GPU
- Final framebuffer copied back per sample for progressive display

## CPU vs GPU Rendering

Xenon provides two independent rendering backends that implement the same algorithm:

| | CPU Backend | GPU Backend |
|---|---|---|
| **Architecture** | Tiled integrator — one path per pixel, processed in 32×32 tiles for cache locality | Wavefront kernels — rays sorted into stage-specific GPU kernels (raygen, intersect, shade, shadow) |
| **Parallelism** | Thread pool distributing tiles across all CPU cores | CUDA blocks/threads processing thousands of rays in parallel |
| **BVH Traversal** | SSE-accelerated AABB intersection with dual-child ILP | Scalar traversal in CUDA device code |
| **When to use** | No GPU available, debugging, small scenes | Large scenes, high sample counts, interactive preview |

Both backends produce the same image for a given scene and sample count. The CPU backend is always available; the GPU backend requires an NVIDIA GPU with CUDA support.

## Building

### Dependencies

- **C++17** compiler (GCC 10+ or Clang 12+)
- **CMake** 3.20+
- **GLFW 3** — windowing and OpenGL context (`libglfw3-dev` on Debian/Ubuntu)
- **OpenGL** — live preview display
- **CUDA Toolkit** (optional) — GPU backend. If not found, Xenon builds with CPU-only support.

All other dependencies (stb_image_write for PNG output) are vendored in the repository.

### Build Steps

```bash
git clone https://github.com/your-username/xenon.git
cd xenon

# Release build (recommended)
./xenon.sh --build

# Debug build
./xenon.sh --build --debug
```

Or manually with CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

To disable CUDA even when a toolkit is available:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DXENON_ENABLE_CUDA=OFF
```

## Usage

```bash
# Render a scene with live preview window
./xenon.sh --scene cornell_box --samples 256

# Render to a file (no window)
./xenon.sh --scene cornell_box --samples 512 --output render.png --no-display

# Render with the GPU backend
./xenon.sh --scene cornell_box --samples 256 --backend cuda

# Custom scene file
./build/release/xenon --scene scenes/lucy/lucy.xenon --samples 128
```

### Scene Format

Scenes are defined in `.xenon` text files:

```
config 800 600 256           # width height samples
camera 0 5 15 0 5 0 40      # eye_xyz target_xyz fov
matfile floor.mat            # load a material file
mesh scenes/model.obj        # load an OBJ mesh
sphere 0 1 0 1.0 glass       # primitive: center radius material
light 0 34 255 182 120 0.5   # area light: triangle_id emission
sun 1 -1 0.5 3.0 3.0 2.8    # directional light: direction rgb
```

Materials use a separate `.mat` format with physically meaningful parameters (albedo, roughness, metallic, IOR, transmission, etc.).

## Implementation Overview

```
src/
├── main.cpp                  # Entry point, argument parsing, render loop
├── math/                     # Vec3 (SSE), Mat4, Ray, SIMD utilities
├── geometry/                 # Triangle mesh, analytic primitives, BVH
│   ├── blas.cpp              # Bottom-level BVH (per-mesh, binned SAH)
│   ├── tlas.h                # Top-level BVH over instances
│   ├── primitives.h          # Sphere, box, plane, disk, quad
│   └── transform.h           # Affine transforms for instancing
├── material/                 # Material system and BSDF evaluation
│   ├── material.h            # Material struct and .mat parser
│   └── bsdf.h                # Diffuse, GGX, glass, subsurface BSDFs
├── camera/                   # Perspective camera with thin-lens model
├── render/                   # CPU rendering pipeline
│   ├── cpu_tiled.cpp          # Tiled path integrator (main CPU path)
│   ├── wavefront.cpp          # Staged wavefront pipeline (CPU reference)
│   ├── thread_pool.h          # Lock-free work-stealing thread pool
│   └── swapchain.h            # Triple-buffer swapchain for display
├── cuda/                     # GPU rendering pipeline
│   ├── cuda_renderer.cu       # CUDA wavefront kernels
│   ├── gpu_scene.cu           # Scene upload to device memory
│   ├── gpu_bsdf.cuh           # Device-side BSDF evaluation
│   └── gpu_traverse.cuh       # Device-side BVH traversal
├── scene/                    # Scene loading and management
│   ├── scene.h                # Scene container (geometry, materials, lights)
│   └── scene_file.cpp         # .xenon file parser
└── display/                  # GLFW window and OpenGL display
```

### Key Design Decisions

- **Two-level BVH (TLAS/BLAS):** Each mesh or primitive group gets its own bottom-level BVH. A top-level BVH indexes instances with transforms. This supports instancing without duplicating geometry data.
- **Binned SAH construction:** BVH nodes are split using the surface area heuristic evaluated over 12 candidate bins per axis, balancing build speed against traversal quality.
- **Wavefront pipeline:** The renderer processes paths in discrete stages (generate, intersect, shade, shadow) rather than tracing complete paths per thread. This maps cleanly to GPU kernel launches and keeps the CPU and GPU implementations structurally identical.
- **Material classification:** Each hit is classified into exactly one BSDF lobe (diffuse, microfacet reflection, microfacet transmission, delta reflection, delta transmission, subsurface) before shading. The CPU backend processes all lobes inline; the GPU backend could sort by lobe type to reduce warp divergence.

## Performance

Performance depends on scene complexity, material types, and sample count. Approximate single-frame times on representative hardware:

| Scene | Triangles | Samples | CPU (16-thread) | GPU (CUDA) |
|---|---|---|---|---|
| Cornell Box | ~30 | 256 | ~2s | ~0.3s |
| Cow | ~6k | 256 | ~8s | ~1s |
| Lucy | ~100k | 256 | ~25s | ~3s |

*Times are approximate and depend on hardware, resolution, and bounce depth.*

The CPU backend uses a tiled integrator optimized for cache locality. Each 32×32 tile is processed by a single thread, keeping the working set small. BVH traversal uses SSE intrinsics for dual-AABB intersection, achieving roughly 2× throughput over scalar code through instruction-level parallelism.

## Running Tests

```bash
cmake --build build --parallel $(nproc)
cd build && ctest --output-on-failure
```

Tests cover vector math, random sampling distributions, BSDF energy conservation, and wavefront pipeline correctness.

## Future Improvements

- Texture mapping (albedo, normal, roughness maps)
- HDRI environment maps with importance sampling
- Denoising (OIDN or wavelet-based)
- Adaptive sampling based on per-pixel variance
- GPU-side BVH construction and refitting
- Volumetric scattering (participating media)
- Bidirectional path tracing / photon mapping

## License

This project is provided for educational and portfolio purposes.
