# Material energy accounting

Xenon keeps the original single-scattering model by default. Two independent controls add a tested material subset:

```
multipleScattering 1
diffuseRoughness 0.6
```

`multipleScattering` accepts 0 or 1 and is restricted to opaque, isotropic materials with IOR in [1,3]. Unsupported combinations fail during material preparation; anisotropic or transmissive multiple-scattering compensation is not silently approximated. Metallic blends and base-color/roughness/metallic textures work within this subset. `diffuseRoughness` accepts [0,1], is independent of GGX specular roughness, and defaults to zero (Lambertian).

## GGX compensation

The base specular lobe uses the existing GGX NDF and height-correlated Smith visibility. A reciprocal Kulla–Conty compensation lobe returns the directional energy missing from repeated microfacet scattering:

`f_ms = F_ms * (1-E(wo)) * (1-E(wi)) / (pi * (1-E_avg))`.

The color factor is `F_ms = F_avg² * E_avg / (1-F_avg*(1-E_avg))`. Conductors use the cosine-weighted Schlick average; dielectrics use numerically integrated exact Fresnel. This is a microfacet multiple-scattering approximation, not a random walk through the microscopic surface.

The checked-in directional-albedo table is generated from **Xenon's actual GGX implementation** by [generate_energy_lut.cpp](../scripts/generate_energy_lut.cpp), with 8,192 deterministic Hammersley/VNDF samples per cell. Its axes are 33 perceptual roughness samples, 65 view cosines, and 33 dielectric IOR samples concentrated near index matching. A separate channel integrates unit Fresnel. Average albedos integrate the piecewise-linear cosine interpolant exactly, so the reciprocal compensation term normalizes consistently with the lookup. Lookup interpolation and the finite numerical table produce small, measurable errors.

To regenerate:

```sh
c++ -O3 -std=c++20 -mavx2 -mfma -fopenmp -Isrc scripts/generate_energy_lut.cpp -o /tmp/xenon-energy-lut
OMP_NUM_THREADS=12 /tmp/xenon-energy-lut > /tmp/energy_lut_data.inc
cp /tmp/energy_lut_data.inc src/material/energy_lut_data.inc
```

The same generated values and interpolation functions are compiled for CPU and CUDA. This avoids an independently fitted GPU material model.

## Diffuse/specular combination

For an opaque dielectric with zero diffuse roughness, a reciprocal complementary diffuse lobe fills the remaining budget: `f_diff = color * p(wo) * p(wi) / (pi * p_avg)`, where `p = 1-E_specular` and `p_avg` is its projected hemispheric average. The specular energy includes single and compensated multiple scattering. This combination preserves a white substrate's energy and reciprocity. It is still an approximate layer: refraction into the substrate and detailed inter-layer transport are not represented.

When diffuse roughness `r` is nonzero **and** GGX compensation is enabled, Xenon blends `(1-r)` of that complementary Lambert lobe with `r` of a symmetrically attenuated EON lobe: `p(wo)*p(wi)*f_EON(color,r)`. Both terms are reciprocal and energy bounded. This approximate rough layer can lose energy even for white color; it is not the exact physical OpenPBR rough-diffuse substrate. This limitation is separate from the EON component's own white-energy preservation.

The rough diffuse component implements the EON model of Portsmouth, Kutz and Hill: Fujii Oren–Nayar plus its analytical reciprocal energy-compensation term. The polynomial directional-albedo approximation from the paper is used. Its color parameter is the single-scattering reflectance; colored rough materials can lose energy through repeated absorption. EON remains energy preserving for white reflectance. Sampling currently uses cosine hemisphere sampling, which is unbiased but has higher grazing-angle variance than the paper's CLTC importance sampler.

## Reciprocal approximate clearcoat

```
clearcoat 0.7
clearcoatRoughness 0.2
clearcoatIor 1.5
```

The coat weight and roughness are in [0,1], and its IOR is in [1,3]. Coats currently require an opaque substrate; coated transmission fails validation. The top interface is an isotropic, compensated GGX reflection lobe. Coverage `w` combines it with the base as:

`f = w*f_coat + [(1-w) + w*(1-E_coat(wo))*(1-E_coat(wi))]*f_base`.

Symmetric entry/exit attenuation preserves reciprocity and bounds the total reflected energy by one when the base is energy bounded. It omits repeated reflections between the coat and substrate, substrate refraction and coating absorption, so the complete layered model can lose energy in a white furnace. For example, a smooth white mirror with coat coverage 0.7, coat IOR 1.5 and view cosine 0.3 returns approximately 0.8848; this matches the implemented analytical `1-w*F*(1-F)` expression. It must not be presented as an exact lossless physical coating or an exact Blender Coat model.

Sampling includes the coat, base, and their compensation lobes in the returned full mixture PDF. Coincident ideal-reflection lobes are combined into one discrete event, including both contributions and both probabilities.

## Validation

`energy_test` independently integrates the implemented BRDF over incoming directions, tests white furnaces across roughness/view angle/IOR, checks reciprocal components, checks colored absorption bounds, compares Monte Carlo energy to quadrature, and compares sampled direction bins to integrated PDFs. It preserves the original `bsdf_test` single-scattering analytic result `1-ln(2)` at GGX alpha=1. `cuda_bsdf_test` includes the new controls and compares CPU/device evaluations, PDFs and matching random-stream samples.

## Native material study

After exporting the shared Blender studio, create a separate native variant:

```sh
python3 scripts/create_material_study.py \
  --source build/fixtures/hdr_studio/scene.xenon \
  --output build/fixtures/energy_studio
```

The script copies material files, resolves geometry/environment/image paths, and writes `material-study.json` with source dependency hashes and every material change. It enables compensation on eligible nonemissive opaque materials, adds EON rough diffuse to the floor and ceramic, adds a subtle ceramic coat, and gives the closed glass sphere RGB Beer absorption. The original exported comparison source is untouched. This is an intentionally authored Xenon beauty study, not a matched Cycles comparison. Keep its referenced source assets alongside the manifest; regenerate after changing those assets.

In the initial CPU qualification grid, the compensated white-metal error was at most 0.22%, compensated white-dielectric error at most 0.19%, and EON white-furnace error at most 0.027%. These are numerical test results for the documented grid, not guarantees over all possible materials and angles.

References: [Kulla and Conty, Revisiting Physically Based Shading at Imageworks (2017)](https://fpsunflower.github.io/ckulla/data/s2017_pbs_imageworks_slides_v2.pdf), [EON paper and errata (2025, updated 2026)](https://jcgt.org/published/0014/01/06/), [OpenPBR physical structure and layering discussion](https://academysoftwarefoundation.github.io/OpenPBR/).
