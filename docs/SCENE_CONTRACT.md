# Xenon scene contract, version 2

A scene starts with `version 2`. Unknown commands and material keys are errors. Asset paths are relative to the file containing the reference, and may be quoted. Unlike legacy scenes, `matfile "materials/name.mat"` resolves directly relative to the scene file; no directory is prepended.

## Coordinates and geometry

- Distances are meters; the right-handed world uses +Y up. A Blender exporter maps `(x,y,z)` to `(x,z,-y)` and applies Blender's unit scale.
- `mesh "object.obj"` imports positions, explicit normals, UVs and per-face `usemtl` names. OBJ face indices may be positive or negative. OBJ position/normal/UV tuples become separate vertices so hard normals and UV seams survive.
- Optional mesh placement is `tx ty tz uniform_scale rotate_x rotate_y rotate_z`, with degree rotations applied Y, X, Z after scale. The initial exporter bakes arbitrary object transforms and evaluated modifiers into mesh geometry. Native `instance` commands preserve shared geometry and nonuniform affine transforms (see below).
- Normals are outward, transformed with the inverse transpose. Texture tangents derive from the triangle's position and UV gradients, transform as vectors, and retain the sign of mirrored UVs. Degenerate UV gradients have no normal/bump perturbation. Tangents are per-triangle derivatives, not MikkTSpace; smooth tangent-frame differences from Blender are possible.
- UV `(0,0)` is the image's bottom-left. Every triangle using an image-textured material must supply UVs. Analytic primitives accept constant materials; export a UV mesh for textured primitives.
- Displacement must be applied to exported evaluated geometry. Native displacement, subdivision and motion blur are not implemented.

## Camera and output

```
config 800 600 256
camera eye_x eye_y eye_z target_x target_y target_z vertical_fov_degrees
camera_up 0 1 0
lens aperture_diameter_meters focal_distance_meters
```

The default camera is a pinhole. The optional lens samples a circular aperture uniformly; focus distance is along the optical axis. Output is linear RGB radiance before the selected display transform. PNG/display transforms do not change stored transport radiance.

## Materials and textures

```
version 2
name "paint"
type principled
baseColor 1 1 1
roughness 1
metallic 0
ior 1.5
transmission 0
emission 0 0 0
opacityMode mask
opacity 1
alphaCutoff 0.5
baseColorTexture "../textures/albedo.png" srgb
roughnessTexture "../textures/roughness.png" linear r
normalTexture "../textures/normal.png" linear
normalScale 1
```

`type principled` selects Xenon's existing GGX/diffuse mixture. It is **not** an exact implementation of Blender Principled, OpenPBR or a layered multiple-scattering BSDF. The OpenPBR model is a design reference for future material work, not a claim of conformance.

Parameters:

| Field | Meaning |
|---|---|
| `baseColor` | Linear RGB reflectance, each channel in [0,1]. |
| `roughness` | Perceptual GGX roughness in [0,1]; alpha = roughness squared. |
| `diffuseRoughness` | Independent EON rough diffuse control in [0,1]; zero is Lambertian. See [energy-model limitations](MATERIAL_ENERGY.md) for compensated-layer behavior. |
| `multipleScattering` | 0 (default) or 1: enable isotropic opaque GGX compensation and reciprocal diffuse energy accounting; IOR must be in [1,3]. |
| `clearcoat`, `clearcoatRoughness`, `clearcoatIor` | Optional reciprocal, energy-bounded approximate clearcoat on opaque substrates. Defaults 0, 0.03, 1.5; see [coat limitations](MATERIAL_ENERGY.md). |
| `metallic` | Dielectric/conductor mixture weight in [0,1]. |
| `ior` | Positive relative dielectric IOR; determines normal-incidence Fresnel. |
| `transmission` | Dielectric transmission weight in [0,1], distinct from opacity. |
| `anisotropy` | Signed GGX anisotropy in [-1,1]. The BSDF currently uses its generated normal basis, not the UV tangent orientation. |
| `emission`, `emissionTwoSided` | Nonnegative linear RGB emitted radiance. `emissionTwoSided 0` (default) emits only from the front; `1` emits equal radiance from both faces. Triangle emitters participate in area-light sampling with consistent sidedness in NEE and MIS. |
| `opacityMode`, `opacity`, `alphaCutoff` | `mask` (default) accepts opacity × sampled map channel ≥ cutoff. `blend` uses that product as fractional surface coverage through stochastic acceptance. Camera, continuation and shadow rays share the same rule. Opacity and cutoff are in [0,1]. |
| `normalScale` | Scale of tangent-space normal X/Y before renormalizing. OpenGL +Y convention. |
| `bumpScale` | Height scale in scene units. Central differences and the dual surface basis convert map height gradients to a perturbed normal. |
| `textureLod` | Nonnegative explicit mip level, with fractional trilinear interpolation; default 0. Automatic ray differentials/anisotropic texture filtering are not implemented. |

Texture fields: `baseColorTexture`, `roughnessTexture`, `metallicTexture`, `emissionTexture`, `opacityTexture`, `normalTexture`, `bumpTexture`. Syntax is:

```
<field> "relative/path" srgb_or_linear [r_or_g_or_b_or_a]
```

Color and scalar textures multiply their material constants. Set a constant to one when a texture should provide the whole value. Scalar maps default to the red channel. Color maps read RGB; a channel selector applies only to scalar maps. Alpha channels always remain linear. Opacity does not implicitly read the base-color texture's alpha; reference that image explicitly using `opacityTexture ... linear a`.

Base-color and emission maps permit `srgb` or `linear`. Data maps require `linear`. LDR RGB sRGB decoding occurs before filtering. Image rows, UV orientation, bilinear repeat wrapping (including negative coordinates), channel selection and trilinear mip sampling have matching CPU/CUDA implementations. Mips use area-weighted box averaging in linear light, including odd resolutions. Normal mip averages are decoded and renormalized at the hit. Reflectance/scalar factors are bounded to [0,1]; emission retains HDR values.

PNG, JPEG, TGA, BMP and the other formats supported by pinned stb_image v2.30 are accepted. PNG retains 16-bit channel precision. Radiance HDR images require `linear`; EXR and UDIM tiles are not supported. Images are cached by canonical path plus color space and uploaded once per unique decoded image.

Unsupported material behavior fails explicitly: unknown keys/types; coated transmission; nonzero version-2 subsurface; unsupported nondefault emission temperature; and legacy inline `specular` values other than 0.5. The legacy `subsurface` lobe is a diffuse proxy, not volumetric subsurface transport. See the absorption and material-energy sections for implemented extensions and their limitations; these do not establish OpenPBR conformance.

## Validation and comparisons

`texture_test` covers color-space decoding, alpha channel semantics, wrap/orientation/filtering, mip averages, OBJ seams, scalar map channels, normal-map handedness, bump evaluation, nonuniform instances, explicit unsupported cases and actual textured rendering. `cuda_texture_test` runs device hit/visibility/material evaluation and compares linear CPU/CUDA renders.

Sources: [stb_image v2.30 pinned source](https://github.com/nothings/stb/blob/f0569113c93ad095470c54bf34a17b36646bbbb5/stb_image.h), [glTF texture/material conventions](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html), [OpenPBR specification](https://academysoftwarefoundation.github.io/OpenPBR/).

## Native instances

`instance mesh_index m00 m01 m02 tx m10 m11 m12 ty m20 m21 m22 tz` places a previously declared mesh with a row-major affine 3×4 matrix. Nonuniform scale, shear and reflection are supported; singular/nonfinite matrices fail validation. The presence of any `instance` command suppresses automatic identity placements for every mesh asset. `instances_explicit` also suppresses automatic placements, including when no instances are desired. Primitive declarations retain their own identity group. The source mesh and its acceleration structure are shared by placements, and emissive triangles receive per-instance world-space areas and stable IDs.

## Illumination

- Emission and environment RGB are scene-linear radiance. No transport-side clamping or exposure is applied.
- `pointlight position_xyz intensity_rgb` is an ideal isotropic point source with radiant intensity in W/sr; irradiance follows inverse-square falloff. Blender isotropic power converts to intensity as power/(4π).
- `spotlight position_xyz outward_axis_xyz intensity_rgb inner_half_angle_degrees outer_half_angle_degrees` uses smoothstep falloff in cosine space between its cones. Angles satisfy 0 ≤ inner ≤ outer < 90. This is an explicit Xenon spot convention, not a claim of exact Blender finite-source equivalence.
- `sunlight direction_toward_source_xyz color_rgb irradiance_scale` is an ideal directional source. Its spectrum times scale is perpendicular irradiance, with no inverse-square falloff. Finite sun disks are unsupported.
- `environment "path.hdr" intensity rotation_degrees` reads scene-linear latitude-longitude radiance; longitude is `atan2(z,x)`, +Y is the north pole, and +X has texture u=0.5 before rotation. Positive rotation shifts texture longitude positively. Horizontal filtering repeats and vertical filtering clamps at the poles. Importance weights combine 95% pixel luminance × exact pixel solid angle and 5% uniform solid angle, preserving support under bilinear filtering. Both gradient and mapped environments participate in NEE with MIS on path misses.

The integrators sample one emitter from each active class (triangle, directional, point/spot, environment) per non-delta scattering vertex. Triangle selection uses the power-weighted distribution described below; its selection PMF is included in MIS. Ideal directional and point/spot lights use a discrete selection PMF and no competing BSDF-hit strategy.

## Participating media and absorption

```
volume_box min_x min_y min_z max_x max_y max_z sigma_a_r sigma_a_g sigma_a_b sigma_s_r sigma_s_g sigma_s_b g
```

A box defines invisible, axis-aligned boundaries of a homogeneous medium. Coefficients are nonnegative inverse meters and `abs(g) < 0.999`. The phase function uses Henyey–Greenstein; positive g scatters preferentially in the incoming propagation direction. RGB free-flight sampling uses a uniformly selected channel and the full channel-mixture density. Pure absorption is deterministic Beer–Lambert attenuation. Shadow connections include extinction through every intersected medium. Overlapping box interiors fail validation; heterogeneous fields are unsupported.

A transmissive material may declare `absorption r g b`, also in inverse meters. Transmission enters or exits its closed surface, while reflection retains the current medium. Up to eight enclosing interiors are tracked; overflow fails the render instead of silently truncating. `camera_medium "material_name"` specifies a camera initially inside an absorbing object. The outermost default is vacuum. Interfaces still use the existing air/material IOR model; nested interfaces with different non-air IORs are not an exact nested-dielectric model. Geometry must form correctly oriented closed boundaries for interior absorption.

Analytic regression tests cover Beer–Lambert distance attenuation, the single-scattering slab integral, HG normalization and mean cosine, point/spot inverse-square falloff, media shadow attenuation, and equivalent CPU/CUDA/OptiX results. Circular-aperture tests verify uniform lens sampling and focal-plane convergence.

Fractional coverage uses a fresh query seed and a deterministic integer hash of logical instance, primitive and incident side, independent of traversal order. Rejected surfaces do not consume path depth. Analytic sphere/box entry and exit faces have separate decisions. Emissive NEE samples multiply radiance by coverage; an accepted emissive hit retains its raw radiance. This models coverage holes, while `transmission` models scattering through a dielectric interface. `opacity_test`, `cuda_opacity_test`, `optix_opacity_test` verify textured multilayer transmission, transformed shared instances, finite shadows, analytic roots and rejected coplanar masks. `opacity_transport_*` compare actual renders to analytic camera/shadow coverage and an independent area-light integration.

Triangle light selection uses a radiant-power proxy (world-space area × luminance of the constant emission multiplier, multiplied by two for a two-sided emitter), mixed with 5% uniform probability. Texture content affects evaluated emission but is not included in that approximate selection weight. Every triangle retains nonzero support, and the exact stored selection PMF enters both NEE and BSDF-hit MIS. Double-precision CDF accumulation prevents rare lights from disappearing through float rounding; emitter lookup and selection use binary search. Directional and point/spot sources remain uniformly selected within their respective families. `light_sampling_test` covers analytic PMFs, frequencies and 131,072 emitters with extreme brightness contrast; the CUDA test checks identical uploaded probabilities and choices.
