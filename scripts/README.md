# Tools

Run these from the repository root. Use `./xenon.sh` for ordinary rendering.

| Task | Script |
|---|---|
| Format first-party code or check its style | `format.sh` |
| Download a free Blendkit scene | `download_blendkit.py` |
| Import a local scene from the command line | `import_scene.py` |
| Download Blender and benchmark assets | `scene_assets.py` |
| Restore original local mesh assets | `setup_assets.py` |
| Export a Blender scene | `blender_export.py` |
| Import glTF, GLB or OBJ through Blender | `blender_import_scene.py` |
| Convert and bake a comparison source | `blender_prepare_scene.py`, `blender_bake_worker.py` |
| Export a material variant with shared geometry | `blender_material_variant.py` |
| Generate studio and fog fixtures | `create_blender_fixture.py`, `create_blender_volume_fixture.py` |
| Generate the native material study | `create_material_study.py` |
| Compare linear images | `compare_images.py` |
| Benchmark one scene or the shared suite | `benchmark.py`, `qualify_scenes.py` |
| Profile GPU scheduling | `benchmark_gpu_scheduling.py`, `analyze_gpu_profile.py` |
| Generate the material energy lookup table | `generate_energy_lut.cpp` |
| Inspect path-seed statistics | `probe_path_sampling.cpp` |
| Set up the documented WSL OptiX workaround | `setup_optix_wsl.sh` |

Blender scripts run inside Blender with `--python`; ordinary Python tools provide `--help`. The C++ analysis tools have reproduction commands in their associated technical notes. `gen_cornell.py` and generators next to the original scenes are historical utilities, not the current asset installation path.

[Import commands](../docs/BLENDER_IMPORT.md) · [Benchmark commands](../docs/QUALIFICATION.md) · [Asset provenance](../docs/ASSETS.md)
