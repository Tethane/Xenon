# Xenon documentation

| Start here | |
|---|---|
| [Install and run](GETTING_STARTED.md) | Launcher, build options, output files and original scene assets |
| [Render gallery](renders/README.md) | Current images, Cycles comparisons and early renders |
| [Download, import and render](IMPORTING_SCENES.md) | Everyday workflow for downloaded scenes and models |
| [Blendkit scenes](BLENDKIT_SCENES.md) | Laundromat, warehouse and autumn cabins: reproduction and conversions |
| [Blender import](BLENDER_IMPORT.md) | Export `.blend` files, import glTF/OBJ, prepare benchmark scenes |
| [Development](DEVELOPMENT.md) | Consistent code formatting and focused validation |
| [Blendkit and grass](BLENDER_IMPORT.md#blendkit-scenes-and-grass) | Mesh scatter, hair conversion and scene compatibility |
| [Asset setup](ASSETS.md) | Downloads, checksums, attribution and licenses |

| Technical reference | |
|---|---|
| [Scene contract](SCENE_CONTRACT.md) | Coordinates, geometry, cameras, lights, textures and media |
| [Material format](MAT_FORMAT.md) | `.mat` syntax and parameter meanings |
| [Material energy](MATERIAL_ENERGY.md) | BSDF models, validation and approximations |
| [Feature coverage](IMPLEMENTATION_STATUS.md) | Implementation, tests and supported boundaries |
| [OptiX](OPTIX.md) | Traversal architecture, build requirements and WSL runtime |

| Measurements | |
|---|---|
| [Real-scene qualification](QUALIFICATION.md) | Repeated benchmarks, linear image errors and reproduction |
| [GPU scheduling](OPTIX_PERFORMANCE.md) | Controlled experiments and profiler evidence |
| [Sampling](SAMPLING.md) | Seed correlation diagnosis and regression coverage |
| [Raw reports](qualification/) | JSON, timing CSVs, test logs and artifact checksums |
| [Historical logs](archive/README.md) | Preserved early runs; separate from current benchmarks |

Published images live in `renders/` within this directory. New local renders go to the repository's ignored top-level `renders/`. Downloaded assets, Blender exports and build products stay under `build/`.
