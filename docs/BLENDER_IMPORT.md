# Reproducible Blender input

Xenon imports the evaluated output of Blender through a focused exporter. It does not parse arbitrary `.blend` files directly. For the host-side command, start with [download, import and render](IMPORTING_SCENES.md). The direct Blender commands below expose the lower-level exporter; use `--python-exit-code 1` so failures propagate to automation and `--disable-autoexec` before opening downloaded `.blend` files. The recorded qualification uses Blender 4.5.3; focused export, scatter and light checks also pass on Blender 5.2.2.

## First fixture

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/create_blender_fixture.py -- \
  --output build/fixtures/material_studio --width 512 --height 384 \
  --samples 64
```

This creates a textured ceramic sphere, brass ring, glass sphere, beveled plinth, floor, two mesh emitters, and a perspective camera. All source geometry and image maps are generated deterministically in Blender. Outputs include `material_studio.blend`, `scene.xenon`, OBJ geometry, `.mat` files, image textures, `export-report.json`. Generate the matched Cycles reference using the final-suite commands below.

`cycles.exr` and `cycles.pfm` contain scene-linear Rec.709 RGB with a Box pixel reconstruction filter and without denoising, adaptive sampling, radiance clamping, or a display transform. PFM is float32 little-endian with bottom-up rows. `cycles-metrics.json` records Blender version, device, resolution, samples, seed, bounce limit, and render time. CUDA or OptiX reference requests fail if that device is unavailable; they never silently choose CPU.

Use Xenon's `--linear-output` to compare the same scene before tone mapping. Cycles and Xenon use different BSDF implementations and sampling strategies, so matching geometry/material parameters do not imply equal radiance at every material. Separate these model differences from image noise and backend agreement.

## Export an existing source

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background input.blend --python-exit-code 1 \
  --python scripts/blender_export.py -- \
  --output build/exports/input --width 800 --height 600 --samples 256
```

The exporter creates [version 2 scene and material files](SCENE_CONTRACT.md). It preserves evaluated geometry and modifiers, transforms including nonuniform scale and mirrors, split corner normals, UV seams, active render UVs, face material assignments, active camera pose/framing/roll, and supported light/material data. Blender `(x,y,z)` becomes Xenon `(x,z,-y)`; scene units become meters. By default, instances are realized into world-space triangles. Add `--instances` to hash evaluated positions, corner normals, UVs, topology, and material bindings, store each unique mesh once, and export its affine placements. This preserves repeated meshes under nonuniform scale, shear, and reflection; counts are recorded in the export report. Content hashing avoids incorrectly sharing objects whose modifiers or material bindings differ.

Supported surface roots are a restricted Principled BSDF, zero-roughness Diffuse BSDF, single-scattering GGX Glossy/Glass, and Emission. Principled supports base color, metallic, roughness, IOR, transmission, emission, fractional opacity and explicit cutout masks, image tangent-space normals, and standalone image bump. Unsupported nondefault Principled inputs fail clearly. Image inputs use flat mapping, repeat addressing, bilinear filtering, sRGB or linear Rec.709, and the active render UV map. Scalar packed channels through RGB-mode Separate Color/RGB are supported. Direct image Color into a scalar socket is accepted only when all pixels are verified grayscale; otherwise Blender luminance conversion requires baking. Texture-linked IOR/transmission, nonunit normal-map color multipliers, lossy packed HDR conversion, and unsupported graph operations fail explicitly. Procedural graphs, custom groups, arbitrary mixed closures, shader displacement, and unsupported coordinate chains require conversion or baking. The homogeneous volume subset is described below.

Constant worlds, linear equirectangular HDR environment maps, ideal directional suns, normalized zero-radius point lights, perspective pinhole cameras, and circular finite-aperture cameras are supported. Blender area lamps have visibility semantics distinct from mesh emitters; use emissive meshes in the shared source. Cycles mesh Emission and Principled emission radiate from both geometric sides. The exporter writes `emissionTwoSided 1` to preserve this behavior for light hits, direct-light sampling, and MIS; the requirement appears in the export report. Hand-authored legacy Xenon materials retain one-sided emission by default.

For opaque Blender `MULTI_GGX` materials, `--compensated-materials` selects Xenon's energy-compensated isotropic GGX model when IOR is in `[1,3]`. This is the recommended approximation for that supported range; it is explicitly recorded per material and is not an exact implementation of the Cycles closure. Transmission and out-of-range IOR retain the reported single-scattering approximation. The option leaves the Blender source untouched, so its existing Cycles reference remains valid. Opaque single-scattering `GGX` materials retain single scattering.

Create a separate comparison variant by reopening an existing shared source and exporting to a new directory:

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/material_studio/material_studio.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_export.py -- \
  --output build/fixtures/material_studio-compensated --width 512 --height 384 \
  --compensated-materials
```

For large scenes, `blender_material_variant.py` avoids re-exporting geometry. Open the unchanged shared `.blend`, provide `--base-export` and a separate `--output`, and add `--compensated-materials`. It verifies the baseline source SHA-256 and all material bindings, then shares the baseline mesh directory and writes independent materials/textures. Keep the baseline directory available.

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/classroom-converted/comparison.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_material_variant.py -- \
  --base-export build/fixtures/classroom-instances \
  --output build/fixtures/classroom-instances-compensated --compensated-materials
```

For Blender tangent normal strength below one, the exporter writes a derived 16-bit linear RGB PNG with `blue = 1-strength+strength*blue`, then retains `normalScale=strength`. This preserves Blender's interpolation toward the unperturbed normal while using Xenon's glTF-style XY scale. Each derived image and checksum appears in the export report. Point-light radiant intensity and aperture diameter are rescaled consistently with scene units so uniform unit conversion preserves the shared source image.

Failure writes `export-report.json` and removes any stale `scene.xenon` in the destination, so partial material/geometry output cannot be mistaken for a successful scene. Reports identify the object/material and unsupported feature. This explicit subset is not complete support for Blender Principled or arbitrary Blender scenes.

## Blendkit scenes and grass

A Blendkit asset must first be available as a local `.blend` with its textures and dependencies. The [download/import workflow](IMPORTING_SCENES.md) uses official downloads and the host exporter; [scene recipes and provenance](BLENDKIT_SCENES.md) record the original sources. Open a compatible Blender version and retain native mesh instances. A gallery page alone does not establish shader or geometry compatibility.

These scenes exercise different preparation requirements:

| Scene | What to inspect |
|---|---|
| [Japanese Laundromat](https://www.blendkit.com/asset-gallery-detail/fea95d8b-bb8d-4bdd-bd53-e1a6252fa3b7/) | The listing describes a 20,320-polygon interior. Start here; inspect materials, window glass and light types. |
| [Forgotten Industrial Warehouse](https://www.blendkit.com/asset-gallery-detail/9cb181dc-d05a-42c0-880c-5949918ce392/) | The listing describes animation, volumetric lighting and Biome Reader scatter. Choose one frame, resolve scatter dependencies, and inspect the volume representation. Xenon supports homogeneous boxes, not arbitrary fog fields. |
| [Autumn Cabins by the Lake](https://www.blendkit.com/asset-gallery-detail/81285b8a-9df7-4f39-84d7-7256057a6666/) | The source uses Blender 5.2, Biome Reader grass/rock scatter and a compositor setup. Use Blender 5.2 or a compatible newer version; focused exporter tests pass on 5.2.2, but that does not establish arbitrary graph compatibility. The Blender compositor and its color grade are not exported. |

**Mesh scatter is supported; raw hair is not automatically converted.** Object/collection particles and Geometry Nodes mesh instances exposed by Blender's dependency graph can export as triangles. `--instances` keeps matching evaluated meshes shared instead of duplicating every blade or clump. [Blender's instance documentation](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/instances.html) describes why retaining instances saves memory.

Strict export requests the **viewport** dependency graph and records `depsgraph_mode` in its report. It now rejects detected cases that could change render geometry: particle display density below 100%, differing active child counts, differing instancer or modifier viewport/render visibility, and unmuted *Is Viewport* nodes in active Geometry Nodes groups, including nested groups. That node check is conservative; the exporter does not prove whether two branches are equivalent. Validation also runs when the emitter's own mesh is hidden but its instances are visible.

1. Save a separate export copy and select the intended frame. Bake or otherwise resolve simulation/add-on dependencies.
2. Match viewport and render scatter density, child counts, modifier visibility and emitter visibility. Remove render-dependent Geometry Nodes branches or bake their intended render result. Hidden particle and Geometry Nodes prototypes retain their visible placements; ordinary hidden objects, hidden instancers and hidden collection-instance members remain omitted. Render-hidden ancestor collections and active-view-layer exclusions are respected, while an object's other visible collection link can keep it visible. Instanced collections use their own render-visible hierarchy. Do not make an otherwise hidden prototype visible merely to export its placements.
3. Convert raw hair/curves to actual surface geometry with width: ribbons or tubes, then a mesh. Modern curves can use **Curve to Mesh** with a profile. A bare strand converted only to edges still has no triangles for Xenon to intersect.
4. Export with `--instances`; compare object, placement and triangle counts and inspect a small preview against Blender before a long render.

Active legacy particle render modes other than **OBJECT**, **COLLECTION** and **NONE** now fail explicitly, including **HAIR/PATH** strands; modern raw `CURVES` objects remain unsupported. Traditional beveled curves that evaluate to triangle meshes can export. The renderer has no native hair intersections or hair BSDF. Successful export still does not prove that an arbitrary add-on or simulation generated its intended result.

`tests/blender_scatter_test.py` passes on Blender 4.5.3 and 5.2.2. It compares flattened and native exports against Blender's independent **RENDER** dependency graph using a minimal capture engine, including 20 particle placements, Geometry Nodes instances, hidden prototypes, collection visibility/exclusions, multiple collection links, nested collection instances, hidden lights and rejected density/path cases. This check evaluates placement data without running Cycles shading.

The host command's explicit `--prepare` path captures the authored frame through a **RENDER** dependency graph before creating a separate static comparison scene. It can therefore retain the render branch and density without making the source's viewport settings match. Its report records source and realized instance counts, meshes, material conversions and any reset per-ray visibility. Supported material graphs are retained; unsupported material slots receive separate bake atlases. This path still rejects unsupported geometry instead of treating missing hair or scatter as a successful conversion.

[Blender particle object/collection modes](https://docs.blender.org/manual/en/latest/physics/particles/emitter/render.html) and [Curve to Mesh](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/curve/operations/curve_to_mesh.html) cover the source representations. Geometry conversion does not translate every shader: foliage translucency, procedural materials, custom groups and compositor effects still need separate review. Full realization is a fallback when instances cannot be exposed correctly, but may greatly increase memory use.

## Intel Sponza comparison source

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_import_scene.py -- \
  --input build/assets/sponza/main_sponza/NewSponza_Main_glTF_003.gltf \
  --output build/fixtures/sponza-full --width 800 --height 600 \
  --samples 256 --max-texture-size 1024 --instances \
  --camera-name PhysCamera001
```

The original glTF has 3,747,018 indexed triangles and 28 materials. Blender validation removes 14 duplicate indexed faces on second-floor columns, leaving 3,747,004 triangles. The import retains all distinct geometry, including fractional-alpha `dirt_decal` surfaces, and records exact source alpha modes on every imported Blender material, including duplicate names and vertex-color variants. Explicit glTF masks preserve their source cutoff; blend materials use stochastic coverage in Xenon. All-white vertex-color multiplication is simplified only after verifying the values. The optional texture limit reduces images to at most 1024 pixels in both renderer sources and records each change in `import-report.json`; omit that flag to preserve original dimensions.

Both renderers use the resulting `source.blend`, including its authored warm directional sun, cool constant world, and Intel-authored `PhysCamera001` camera. These lighting settings are benchmark choices. The original downloads remain untouched. The earlier `sponza-comparison` fixture explicitly omitted dirt decals before fractional opacity was available; use `sponza-full` for complete geometry. A common +4 EV display exposure makes the shaded courtyard readable without changing linear comparison data.

## Classroom comparison conversion

The unmodified Classroom source contains legacy mixed closures, custom shader groups, procedural image operations, shader displacement, finite light radii, and camera aperture blades. Strict export records these limitations. The frozen qualification used the legacy preparation behavior described below. The current preparation tool preserves more supported materials, captures render-time geometry and bakes additional channels; running it again creates a newly reported comparison source rather than reproducing the frozen source bytes or measurements.

The preparation command is:

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/assets/classroom/classroom/classroom.blend \
  --threads 4 --python-exit-code 1 \
  --python scripts/blender_prepare_scene.py -- \
  --output build/fixtures/classroom-converted --width 800 --height 600 \
  --bake-size 512 --samples 256
```

Create the native instance baseline from that saved comparison source:

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/classroom-converted/comparison.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_export.py -- \
  --output build/fixtures/classroom-instances --instances --width 800 --height 600
```

Use the Cycles reference from the same `comparison.blend`. The evaluated geometry and materials remain identical; only mesh storage and placement representation change.

`comparison.blend` is the source for both renderers. The recorded legacy conversion chose a Principled approximation from the reachable legacy closures, baked the chosen base-color graph into a per-mesh UV atlas, retained evaluated modifiers and geometry normals, and explicitly omitted legacy shader bump/displacement. It retained constant roughness and used a documented conversion value for unsupported linked roughness. Per-instance Object Info variation shared the source-mesh bake; view-dependent inputs were evaluated at bake directions. Its `conversion-report.json` records the chosen closures, evaluated meshes, bake atlases and lighting/camera changes. This is a converted benchmark, not a reproduction of the original Classroom shading.

In the recorded conversion, rectangular area lamps became emissive rectangles with radiance `power/(π × transformed area)`; finite suns and point radii became ideal lights, and depth of field was disabled. The current preparer preserves circular aperture settings and explicitly converts noncircular apertures. [Blender 4.5.3 light code](https://github.com/blender/blender/blob/v4.5.3/intern/cycles/blender/light.cpp) and [area-light evaluation](https://github.com/blender/blender/blob/v4.5.3/intern/cycles/kernel/light/area.h) establish the normalized area-power convention. Inspect the conversion report before comparing a regenerated source.

## Glass, fog, and depth of field

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background --factory-startup --threads 4 --python-exit-code 1 \
  --python scripts/create_blender_volume_fixture.py -- \
  --output build/fixtures/glass_fog --width 512 --height 384
```

This writes `glass_fog/material_studio.blend` and adds a homogeneous fog box and circular finite-aperture camera to the shared studio source. The exporter accepts constant Volume Absorption/Volume Scatter closures or their additive combination on a closed axis-aligned eight-corner box with no surface shader, using one Blender unit per meter. It checks the actual evaluated boundary. Absorption is `(1-color)*density`; scattering is `color*density`, matching Blender's volume-closure convention. Arbitrary volume shapes, heterogeneous fields, mixed anisotropies, and other volume closures fail export.

To render a saved source without repeating the geometry export, use `blender_export.py -- --output ... --reference-only --reference-samples 1024 --cycles-device OPTIX`.

## HDR studio

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background --factory-startup --threads 4 --python-exit-code 1 \
  --python scripts/create_blender_fixture.py -- \
  --output build/fixtures/hdr_studio --width 512 --height 384 \
  --environment-hdr build/assets/studio_hdri/studio_small_09_1k.hdr \
  --environment-strength .4
cp build/fixtures/hdr_studio/material_studio.blend build/fixtures/hdr_studio/source.blend
```

The copy gives the shared HDR source the filename used by the frozen qualification. The final-suite commands reopen it and refresh the export/source checksum. This uses the pinned Poly Haven image in both renderers. The exporter copies scene-linear HDR data and writes an environment with zero rotation: the Blender-to-Xenon axis transform aligns Blender's equirectangular coordinates with Xenon's Y-up mapping. Environment Mapping nodes currently require baking into the image.

The optional integration regressions are `tests/blender_export_test.py` (mirrored winding, evaluated shared meshes, UV rejection) and `tests/blender_import_test.py` (alpha provenance and reference framing). Run either with the pinned Blender using `--background --factory-startup --python-exit-code 1 --python PATH`.

## Reproduce the final five-scene suite

Run asset setup and the baseline creation commands above first. The following commands recreate the shared-source references and final compensated variants without altering their Blender material graphs. Use the pinned Blender executable and configure the [OptiX runtime](OPTIX.md) for this machine before requesting `--cycles-device OPTIX`. Cycles references use seed 1, eight maximum bounces, Box filtering, and no denoising/adaptive sampling/clamping. The qualification manifest independently uses Xenon seed 42.

The three small variants retain exact baseline OBJ bytes. Reopening each saved source before reference/export also makes the recorded source SHA-256 describe the file actually compared.

```sh
xenon_blender=build/tools/blender-4.5.3-linux-x64/blender
for xenon_scene in material_studio hdr_studio glass_fog; do
  xenon_base="build/fixtures/$xenon_scene"
  xenon_source="$xenon_base/material_studio.blend"
  if [ "$xenon_scene" = hdr_studio ]; then
    xenon_source="$xenon_base/source.blend"
  fi
  "$xenon_blender" --background "$xenon_source" --threads 4 --python-exit-code 1 \
    --python scripts/blender_export.py -- --output "$xenon_base" \
    --width 512 --height 384 --seed 1 --bounces 8 \
    --reference --reference-samples 1024 --cycles-device OPTIX
  "$xenon_blender" --background "$xenon_source" --threads 4 --python-exit-code 1 \
    --python scripts/blender_export.py -- --output "$xenon_base-compensated" \
    --width 512 --height 384 --compensated-materials
  cp "$xenon_base/meshes/scene.obj" "$xenon_base-compensated/meshes/scene.obj"
done
```

Render Classroom and Sponza from their unchanged saved sources. The reference paths remain in their original directories:

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/classroom-converted/comparison.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_export.py -- \
  --output build/fixtures/classroom-converted --width 800 --height 600 \
  --reference-only --reference-samples 1024 --seed 1 --bounces 8 --cycles-device OPTIX
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/sponza-full/source.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_export.py -- \
  --output build/fixtures/sponza-full --width 800 --height 600 \
  --reference-only --reference-samples 4096 --seed 1 --bounces 8 --cycles-device OPTIX
```

Create the two large material-only variants. Their `meshes` directories link to the corresponding baseline assets, so preserve those baseline directories:

```sh
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/classroom-converted/comparison.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_material_variant.py -- \
  --base-export build/fixtures/classroom-instances \
  --output build/fixtures/classroom-instances-compensated --compensated-materials
build/tools/blender-4.5.3-linux-x64/blender \
  --background build/fixtures/sponza-full/source.blend \
  --threads 4 --python-exit-code 1 --python scripts/blender_material_variant.py -- \
  --base-export build/fixtures/sponza-full \
  --output build/fixtures/sponza-full-compensated --compensated-materials
```

Every current exporter writes `emissionTwoSided 1` for Blender emission automatically; no face reversal or manual material patch is needed. The compensated variants change only eligible material energy compensation: four materials in each studio, 326 in Classroom, and 44 in Sponza. Geometry, camera, emission sidedness, and source references match their baselines. All source checksums, approximation records, texture provenance, and required renderer features are in `export-report.json`; downloads have their independent checksums in [ASSETS.md](ASSETS.md).

Finally, validate the complete manifest without rendering:

```sh
python3 scripts/qualify_scenes.py --binary build/optix/xenon --dry-run
```

[The suite manifest](../scenes/benchmarks/suite.json) selects all five compensated scenes while retaining the original Cycles references. This qualifies the documented subset and converted Classroom source, not arbitrary Blender shader graphs, heterogeneous volumes, or every possible benchmark scene.
