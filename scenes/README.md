# Scenes

Use `./xenon.sh --list-scenes` from the repository root for available names and setup status.

| Original scene | What it exercises |
|---|---|
| `prim_types` | Analytic primitives and materials; no mesh download |
| `sky_dome`, `outdoor_sun` | Environment and directional lighting |
| `instancing_grid` | Repeated placements and material variation |
| `cornell_box`, `cornell_empty` | Indirect lighting in an enclosed room |
| `lucy`, `cow`, `combined_outdoor` | Meshes, materials and larger scenes |

Original OBJ assets are stored beside their scenes but excluded from Git. [Restore and verify them](../docs/GETTING_STARTED.md#scenes-and-assets) before rendering mesh-based scenes.

The [benchmark manifest](benchmarks/suite.json) points to generated Sponza, Classroom, studio and fog exports under `build/fixtures/`. The launcher also accepts `sponza` and `classroom` as short names. [Benchmark setup](benchmarks/README.md) describes the shared Blender sources and references.

For your own scene:

```sh
./xenon.sh --scene /path/to/scene.xenon --backend optix --samples 512
```

[Scene syntax](../docs/SCENE_CONTRACT.md) · [Blender export](../docs/BLENDER_IMPORT.md)
