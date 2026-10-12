# Download, import and render a scene

Xenon renders `.xenon` scenes. Blender is the bridge from downloaded `.blend`, glTF/GLB and OBJ assets: it evaluates geometry, resolves materials and textures, then exports the supported scene representation. The source download stays intact, and the export includes a report of any unsupported features.

## Choose a source

Keep each download and all of its dependencies in a separate directory under `build/assets/`. Extract archives before importing. A `.gltf` needs its companion `.bin` and textures; an OBJ usually needs its `.mtl` and texture images. A `.blend` may depend on linked libraries, external images, simulation caches or add-ons.

For a free Blendkit asset, use its gallery URL or asset ID:

```sh
python3 scripts/download_blendkit.py \
  https://www.blendkit.com/asset-gallery-detail/fea95d8b-bb8d-4bdd-bd53-e1a6252fa3b7/ \
  --output build/assets/blendkit/laundromat
```

The downloader uses Blendkit's download API and writes `source.blend`, `asset-metadata.json` and `provenance.json`. A repeat invocation verifies an existing original; interrupted downloads resume only when the saved asset/version identity matches. Add `--sha256 HASH` to pin known source bytes. Without that option the reported SHA-256 identifies the downloaded file, rather than comparing it with an upstream checksum. If an asset requires authentication or purchase, obtain it through the official service and import the local file instead. Keep the source's license and attribution with your project. The [Blendkit scene recipes](BLENDKIT_SCENES.md) retain the three downloaded sources' identifiers and checksums.

Other sites may provide direct downloads or require a browser/account. Download the actual scene archive, not the gallery page, and preserve its directory structure. For formats such as FBX or USD, use Blender's appropriate importer, inspect the result, and save a `.blend` before exporting to Xenon.

## Select Blender

Use a version that can open the downloaded file. A scene saved by Blender 5.2 should not be assumed compatible with 4.5. Xenon's import command checks `--blender`, then `BLENDER`, local `build/tools/blender-5.2.*` installations, local `blender-4.5.*`, and finally `blender` on `PATH`. Within each local version family, the newest installed patch is selected. An invalid explicit selection fails instead of trying another executable.

```sh
export BLENDER=/path/to/blender
```

The command disables automatic execution of embedded Python when opening downloaded files. Geometry Nodes still evaluate; external Python add-ons and simulations may need a prepared, saved source in Blender.

## Import a Blender scene

Start with strict export into a new or empty destination:

```sh
python3 scripts/import_scene.py /path/to/scene.blend \
  --output build/imports/my-scene --width 960 --height 540 --samples 512
```

The active Blender camera is used. Evaluated mesh instances are shared by default; `--no-instances` requests flattened geometry. Unspecified dimensions retain the Blender render dimensions and percentage; the default scene sample count is 64 and Blender uses four threads (`--threads N` overrides it). `--compensated-materials` opts into the [documented approximation for eligible opaque MULTI_GGX materials](BLENDER_IMPORT.md). The exporter preserves the supported camera, material, texture and light semantics; it does not silently substitute arbitrary shader graphs.

`import-request.json` records the exact command and the source SHA-256 before and after export. The source is left unchanged. Failed exports retain their diagnostic files, and the host command refuses any nonempty output directory: choose a new destination for a corrected attempt or review and remove the failed output yourself.

If strict export reports unsupported content, read `export-report.json`. In Blender, make a separate export copy, resolve missing dependencies, select a camera, apply or bake the relevant geometry, and simplify or bake unsupported material inputs. The [detailed scene contract](SCENE_CONTRACT.md) and [Blender import reference](BLENDER_IMPORT.md) describe the available subset.

An explicit conversion is available for legacy material graphs:

```sh
python3 scripts/import_scene.py /path/to/scene.blend \
  --output build/imports/my-scene-prepared --prepare --bake-size 512
```

`--prepare` creates a separate `comparison.blend` from the authored frame's render dependency graph. Supported material graphs, UVs and corner normals are retained. Unsupported surface slots are converted individually, with UV-tile or mesh-atlas bakes for selected base color, linked roughness/metallic inputs, coverage, emission and normals where applicable. `conversion-report.json` records the chosen closure, bakes and material/light/camera changes. Layered closure behavior, displacement and view- or instance-dependent shader inputs can change; noncircular apertures and unsupported lights also require reported approximations. This is a limited conversion tool, not a universal compatibility switch; inspect its report and a preview before a long render.

Add `--bake-world` only when you want to replace an unsupported world graph with an approximate 2048×1024 scene-linear HDR environment. The bake uses 32 CPU samples and records its image checksum and changes in `world/world-report.json`. Ray-dependent worlds use their diffuse-reflection lighting branch for every Xenon ray; camera and glossy backgrounds may therefore change. Finite angular resolution can blur small sun discs, and world volumes remain unsupported.

`--volume-proxies` is a separate opt-in with `--prepare`: it replaces supported conversions of otherwise unsupported fog with reported homogeneous boxes. Spatially varying density and non-box boundaries become approximations; read `volumes/volume-report.json` and its retained samples before relying on the result. The default 16³ density/occupancy grid preserves estimated integrated density, not ray-by-ray optical depth. Overlap partitioning sums coefficients and reports any phase-function approximation. Without this flag, unsupported volume graphs or boundaries fail explicitly.

For texture-heavy scenes that exceed available GPU memory, explicitly add `--max-texture-size 1024` (or another positive limit). This option is available for model imports and `.blend` inputs with `--prepare`; strict Blender export rejects it. It resizes images in the derived comparison source and export, records the changes, and leaves original downloads untouched. The option is off by default: choose a limit after checking the scene's texture memory needs, since it reduces image detail.

For deliberate scene adjustments, `--prepare --omit-object 'Exact Object Name'` omits that object's render placements; repeat the flag for additional objects. Unknown names and the active camera are rejected. `--omit-area-lights` omits area lamps instead of converting them into visible emissive rectangles. Both choices are recorded. For example, the Laundromat recipe opens one windowpane whose source disables shadows, and removes an auxiliary ceiling fill lamp. These are scene changes, not an exact translation of Blender's ray-visibility controls.

## Import a 3D model

```sh
python3 scripts/import_scene.py /path/to/model.glb \
  --output build/imports/my-model --width 800 --height 600
```

The same command accepts `.gltf` and `.obj`. This route creates `source.blend` with an authored daylight rig; imported light objects are replaced and recorded in `import-report.json`. An available perspective camera is used, otherwise the import tool frames the bounds from outside, accounting for image aspect ratio. Inspect the framing before rendering at high resolution. For full artistic control, open the model in Blender, set its camera and lights, save a `.blend`, and use the Blender-scene route above.

This small internet example was imported and rendered with Blender 5.2.2 and Xenon OptiX:

```sh
mkdir -p build/assets/duck
curl --fail --location \
  https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Duck/glTF-Binary/Duck.glb \
  --output build/assets/duck/Duck.glb
printf '%s  %s\n' 65bf938f54d6073e619e76e007820bbf980cdc3dc0daec0d94830ffc4ae54ab5 \
  build/assets/duck/Duck.glb | sha256sum --check -
python3 scripts/import_scene.py build/assets/duck/Duck.glb \
  --output build/imports/duck --width 640 --height 480 --samples 256
./xenon.sh --scene build/imports/duck/scene.xenon --backend optix \
  --no-display --samples 1024 --seed 42 --output-dir renders/duck \
  --output beauty.png --linear-output radiance.pfm --metrics metrics.json
```

The tested Duck contains 4,212 triangles and one textured material. It is copyright Sony, 2006, under the SCEA Shared Source License 1.0; retain the [source attribution](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Duck/README.md) and [license](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/LICENSES/SCEA.txt). The checksum above identifies the tested download; if upstream changes it, inspect the new version before importing.

## Render and inspect

```sh
# A quick preview
./xenon.sh --scene build/imports/my-scene/scene.xenon \
  --backend optix --samples 64

# Final image, linear radiance and render metrics
./xenon.sh --scene build/imports/my-scene/scene.xenon \
  --backend optix --no-display --samples 2048 --seed 42 \
  --output-dir renders/my-scene --output beauty.png \
  --linear-output radiance.pfm --metrics metrics.json
```

Choose `cpu` or `cuda` if desired; an unavailable backend fails explicitly. The preview and PNG use exposure, Reinhard tone mapping and sRGB. Use `--exposure EV` to adjust the display; it does not change the linear PFM. Closing an unfinished preview returns status 2.

Check camera framing, mesh/instance counts, textures, light placement and missing geometry at low sample counts. Increase samples once those are correct. More samples reduce sampling noise; they cannot restore omitted objects or unsupported shader behavior.

## Grass, hair and procedural scenes

Mesh blades or clumps scattered as object/collection particles or Geometry Nodes instances can become Xenon triangle instances. Raw hair paths need conversion to ribbons or tubes with actual faces; edges alone are not renderable surfaces. Strict export rejects unsupported particle path modes and detected viewport/render differences, including reduced particle density, mismatched child counts and active Geometry Nodes *Is Viewport* nodes. Review the [scatter preparation notes](BLENDER_IMPORT.md#blendkit-scenes-and-grass) for the current checks and limitations.

A typical mesh-and-texture scene is the easiest path. Arbitrary production `.blend` files may also contain hair shaders, heterogeneous volumes, complex material groups, motion blur or compositor effects that Xenon does not reproduce. Keep the original file and make conversions explicit.

Preparation preserves the original UVs when an unsupported material can be evaluated entirely from a single UV tile. Repeated leaf cards then share one baked texture instead of losing detail in a whole-tree atlas. Geometry-dependent graphs use mesh atlases; inspect foliage coverage in a preview, and increase `--bake-size` when needed.
