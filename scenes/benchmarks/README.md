# Generated benchmark suite

The real-scene source assets and generated meshes stay outside version control.
Run [asset setup](../../docs/ASSETS.md) and the commands in
[Blender import](../../docs/BLENDER_IMPORT.md) to create:

- `build/fixtures/material_studio/scene.xenon`: deterministic textured Blender studio.
- `build/fixtures/sponza-full/scene.xenon`: complete Intel Sponza geometry, retained fractional dirt decals, and documented texture reduction.
- `build/fixtures/classroom-converted/scene.xenon`: explicitly converted and baked Classroom.
- `build/fixtures/classroom-instances/scene.xenon`: the same converted Classroom with shared mesh assets and affine placements.
- `build/fixtures/hdr_studio/scene.xenon`: the studio illuminated by the pinned Poly Haven HDR environment.
- `build/fixtures/glass_fog/scene.xenon`: glass, homogeneous fog, and finite-aperture depth of field.

Every generated scene has an export report. Comparison conversions also have an
import/conversion report and a shared `.blend` source for Cycles and Xenon.

The durable qualification manifest is [`suite.json`](suite.json). After all
requested scenes and their matched Cycles references exist, run:

```sh
LD_LIBRARY_PATH="$PWD/build/optix-runtime:/usr/lib/wsl/lib" \
  python3 scripts/qualify_scenes.py --binary build/optix/xenon \
  --samples 128 --repeats 3 --warmups 1
```

Use `--scenes material_studio hdr_studio glass_fog classroom-instances sponza-full`
to select entries, `--backends cpu cuda optix` to select renderers, and `--dry-run`
to check inputs without rendering. Missing scenes/references fail before the
first render. Each scene retains the complete benchmark report; aggregate
summaries and reference metadata are written to
`build/qualification/final/suite-results.json`. The Sponza entry requires the
full-decal conversion's own reference and never substitutes an older conversion.

The manifest uses the final compensated variants below. Follow the
[complete reproduction commands](../../docs/BLENDER_IMPORT.md#reproduce-the-final-five-scene-suite)
to create their baselines, references, and material variants.

| Stable scene ID | Final scene directory under `build/fixtures/` | Original Cycles reference directory |
| --- | --- | --- |
| `material_studio` | `material_studio-compensated` | `material_studio` |
| `hdr_studio` | `hdr_studio-compensated` | `hdr_studio` |
| `glass_fog` | `glass_fog-compensated` | `glass_fog` |
| `classroom-instances` | `classroom-instances-compensated` | `classroom-converted` |
| `sponza-full` | `sponza-full-compensated` | `sponza-full` |

Compensation changes 4, 4, 4, 326, and 44 materials respectively. The large
variants share baseline meshes by directory symlink; the small variants retain
identical OBJ bytes. Blender emission uses `emissionTwoSided 1` in both baseline
and compensated exports. The original references remain valid because the
Blender sources and geometry are unchanged.
