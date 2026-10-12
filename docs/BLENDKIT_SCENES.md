# Blendkit scene imports

These recipes use three real Blendkit downloads saved by Blender 5.2.0 and opened with Blender 5.2.2. The original `.blend` files remain under `build/assets/blendkit/`; explicit preparation writes separate export copies under `build/imports/blendkit/`. This gallery is an import demonstration, separate from the [frozen renderer qualification](QUALIFICATION.md).

## Rendered in Xenon

[View the three finished renders](renders/README.md#downloaded-blender-scenes). Each image comes from Xenon’s OptiX backend, with seed 42 and no denoising or radiance clamping. PNG, linear PFM and metrics are saved together under `renders/blendkit/<scene>/`. The [published render records](renders/blendkit-results.json) include geometry counts, timings and artifact checksums.

| Scene | Resolution | Samples | Placed triangles | Unique triangles | Mesh placements |
| --- | ---: | ---: | ---: | ---: | ---: |
| Japanese Laundromat | 960×540 | 4,096 | 270,296 | 174,976 | 191 |
| Forgotten Industrial Warehouse | 960×960 | 2,048 | 50,542,506 | 4,928,352 | 42,479 |
| Autumn Cabins by the Lake | 800×800 | 2,048 | 328,137,426 | 6,969,778 | 45,799 |

Shared mesh instances keep repeated grass and trees from duplicating their triangle storage. Placed triangle counts include every placement; unique counts describe stored geometry.

Measured on the RTX 4060 Laptop GPU. These are individual gallery runs; total time includes scene loading, acceleration construction, upload and output.

| Scene | Path tracing | Total process |
| --- | ---: | ---: |
| Japanese Laundromat | 131.98 s | 149.41 s |
| Forgotten Industrial Warehouse | 129.60 s | 412.33 s |
| Autumn Cabins by the Lake | 105.53 s | 227.92 s |

## Sources and attribution

| Scene | Author | Download bytes | SHA-256 |
| --- | --- | ---: | --- |
| [Japanese Laundromat](https://www.blendkit.com/asset-gallery-detail/fea95d8b-bb8d-4bdd-bd53-e1a6252fa3b7/) | 3dhub | 73,232,174 | `7a2d9eb9f8394132745c681ac8e37e4633d27acf12adf3c5d24d974627ad5089` |
| [Forgotten Industrial Warehouse](https://www.blendkit.com/asset-gallery-detail/9cb181dc-d05a-42c0-880c-5949918ce392/) | Ibrohim Toxirov | 495,186,237 | `125591d4b6d465c2f52a89f8fd82b46a0d55a94f8821043f28e51896df6745c4` |
| [Autumn Cabins by the Lake](https://www.blendkit.com/asset-gallery-detail/81285b8a-9df7-4f39-84d7-7256057a6666/) | Ibrohim Toxirov | 305,471,294 | `20b02a8aa6d38557e1a0b4a215e1f1cbd87ee48b77587b1dece5eca3a9496f96` |

The public asset metadata marks all three licenses `royalty_free`; this is not a CC0 designation. Keep each asset's author and license records, and follow its linked asset terms. Sources are downloaded separately rather than redistributed in this repository. [blendkit-assets.json](blendkit-assets.json) records asset base IDs, version IDs and the exact tested source checksums. The downloader also writes local `asset-metadata.json` and `provenance.json` files. A checksum mismatch stops the pinned download; inspect any upstream version change before updating a recipe.

## Run the recipes

Run these commands from the repository root. Set `BLENDER` to Blender 5.2.2 if it is not installed under `build/tools/blender-5.2.2-*/blender` or on `PATH`. The [host import command](IMPORTING_SCENES.md) documents discovery, dependencies and strict import. Each export destination must be new or empty; failed imports retain their diagnostics and require a new destination or deliberate cleanup before retrying.

These recipes explicitly request `--prepare --bake-world`. Review `conversion-report.json`, `world/world-report.json` and `export-report.json` before a long render. Unsupported content can still stop preparation. `--compensated-materials` is an explicit approximation for eligible opaque MULTI_GGX materials. The original active camera and source frame are used; no replacement lighting rig is added to `.blend` inputs.

Run these large imports and renders sequentially on a 16 GB host. Source-image estimates below exclude new material-bake atlases and geometry; final memory use can be higher. Keep the whole export directory, including the images referenced by `comparison.blend`.

### Japanese Laundromat

The source opens on frame 93. The output uses its wide camera framing at 960×540. Its active-material image estimate is 2,862.2 MiB as RGBA float32 with mip levels, so this recipe retains source texture dimensions. This composition deliberately omits the auxiliary ceiling area light and opens the single shadow-disabled windowpane `Plane.011`. Window frames, the other glass pane, camera, sun and environment remain. The area lamp would otherwise become a large visible emissive rectangle; Xenon does not reproduce Blender's per-ray visibility switches.

```sh
python3 scripts/download_blendkit.py fea95d8b-bb8d-4bdd-bd53-e1a6252fa3b7 \
  --output build/assets/blendkit/laundromat \
  --sha256 7a2d9eb9f8394132745c681ac8e37e4633d27acf12adf3c5d24d974627ad5089
python3 scripts/import_scene.py build/assets/blendkit/laundromat/source.blend \
  --output build/imports/blendkit/laundromat --prepare --bake-world \
  --bake-size 512 --compensated-materials --omit-area-lights --omit-object Plane.011 \
  --width 960 --height 540 --samples 4096
./xenon.sh --scene build/imports/blendkit/laundromat/scene.xenon \
  --backend optix --no-display --samples 4096 --seed 42 \
  --output-dir renders/blendkit/laundromat --output beauty.png \
  --linear-output radiance.pfm --metrics metrics.json
```

### Forgotten Industrial Warehouse

The source opens on frame 1 with square camera framing.

The source inventory reports a missing displacement image, `PX_Hippophae_rhamnoides_Moss_01_displacement.png`. Preparation already explicitly omits shader displacement; it does not restore that missing detail. Blender also reports an existing dependency cycle through the Ivy object's `IVY_Generator` modifier and its instance collection. The snapshot uses Blender's actual render dependency graph, but that source warning means ivy placement needs visual inspection before claiming scene fidelity. Stale linked-library paths alone did not indicate missing node groups: their loaded datablocks were present.

The Warehouse recipe explicitly limits derived textures to 1024 pixels on their longest side. The source image inventory estimated 13,220.8 MiB for active-material images as RGBA float32 with mip levels, exceeding this machine's 8 GB GPU before geometry or render buffers. A 1024 limit lowers that image estimate to 2,190.5 MiB; 2048 would still require 8,100.8 MiB. These are inventory estimates, not measured renderer allocations. The resized comparison source and export record the change, and the original download remains untouched. Fine texture detail is reduced.

```sh
python3 scripts/download_blendkit.py 9cb181dc-d05a-42c0-880c-5949918ce392 \
  --output build/assets/blendkit/warehouse \
  --sha256 125591d4b6d465c2f52a89f8fd82b46a0d55a94f8821043f28e51896df6745c4
python3 scripts/import_scene.py build/assets/blendkit/warehouse/source.blend \
  --output build/imports/blendkit/warehouse --prepare --bake-world \
  --bake-size 512 --max-texture-size 1024 --compensated-materials \
  --width 960 --height 960 --samples 2048
./xenon.sh --scene build/imports/blendkit/warehouse/scene.xenon \
  --backend optix --no-display --samples 2048 --seed 42 \
  --output-dir renders/blendkit/warehouse --output beauty.png \
  --linear-output radiance.pfm --metrics metrics.json
```

### Autumn Cabins by the Lake

The source opens on frame 5. Start with its square camera framing at 800×800. This recipe additionally opts into measured homogeneous fog proxies; it does not preserve the original fog's spatial density or exact boundaries.

Autumn’s repeated leaf cards retain their original `automap` UVs. Their UV-only color-ramp material is baked once into a reusable 512×512 tile, including opacity and tangent normals. Baking the whole tree into a small atlas would lose leaf coverage; the importer now avoids that for supported UV-only graphs. Object-dependent graphs still require mesh-specific baking.

The recipe also explicitly limits derived textures to 1024 pixels. The active-material image estimate drops from 6,314.2 MiB to 1,968.1 MiB, leaving room on the 8 GB GPU for the large scattered geometry, acceleration structures and render buffers. These source inventory estimates exclude the replacement 2048×1024 world map (about 42.7 MiB as RGBA float32 with mips) and are not measured GPU allocations. Image resolution is reduced only in the derived scene; original files remain intact.

```sh
python3 scripts/download_blendkit.py 81285b8a-9df7-4f39-84d7-7256057a6666 \
  --output build/assets/blendkit/autumn-cabins \
  --sha256 20b02a8aa6d38557e1a0b4a215e1f1cbd87ee48b77587b1dece5eca3a9496f96
python3 scripts/import_scene.py build/assets/blendkit/autumn-cabins/source.blend \
  --output build/imports/blendkit/autumn-cabins --prepare --bake-world --volume-proxies \
  --bake-size 512 --max-texture-size 1024 --compensated-materials \
  --width 800 --height 800 --samples 2048 --threads 2
./xenon.sh --scene build/imports/blendkit/autumn-cabins/scene.xenon \
  --backend optix --no-display --samples 2048 --seed 42 \
  --output-dir renders/blendkit/autumn-cabins --output beauty.png \
  --linear-output radiance.pfm --metrics metrics.json
```

For a quick check, run the render command with `--samples 64` and a separate output directory such as `build/blendkit/previews/laundromat`. Check framing, texture coverage and scatter before rendering the final image. Select `cpu` or `cuda` explicitly when appropriate; an unavailable backend does not silently fall back.

## What the conversion means

These are conversions of authored scenes, not claims of pixel agreement with the original Blender renders. The source scenes contain shader and compositor graphs outside Xenon's strict export subset. Preparation records its choices in the derived source and conversion report; arbitrary layered materials, shader displacement and postprocessing may change. Native triangle instances retain supported scatter efficiently, while raw hair paths and unsupported procedural geometry still require conversion to surfaces. The [Blender import reference](BLENDER_IMPORT.md#blendkit-scenes-and-grass) describes the scatter checks.

The optional world bake evaluates an approximate diffuse-reflection lighting branch into a 2048×1024 scene-linear HDR image with 32 CPU samples. A ray-dependent world becomes one environment for every ray, so camera/glossy backgrounds can change. Finite angular resolution can filter small bright sources. World volumes are unsupported. The world report records the source world name, changes and image checksum.

Autumn's separate `--volume-proxies` option measures density on a 16³ grid of cell centers within each evaluated volume boundary's axis-aligned bounds. It multiplies the measured interior mean density by estimated occupied volume fraction, preserving estimated integrated density in a homogeneous box. This does **not** preserve individual rays' optical depth. Overlapping boxes become disjoint cells with summed extinction/scattering; different phase anisotropies use a reported scattering-weighted approximation. `volumes/volume-report.json`, `samples_*.json` and procedural-density `density_*.exr` files retain measurements and changes. Emissive volume, combined surface/volume and unsupported density-context graphs fail. Warehouse's supported homogeneous volume does not need this option.

Xenon's output PNG uses its own exposure, Reinhard tone mapping and sRGB conversion. Blender compositor effects and its display transform are not reproduced. The linear PFM and `metrics.json` accompany each image so radiance, settings and renderer timing remain inspectable. New gallery renders do not imply additional CPU/CUDA/OptiX speedup or Cycles parity qualification.
