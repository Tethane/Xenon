# `.mat` Material File Format

Material files define physically-based material properties for the Xenon renderer.

## Syntax

Plain text, one key-value pair per line. Comments start with `#`.

```
# Example material
name "GlossyGold"
type "principled"
baseColor 0.944 0.776 0.373
roughness 0.15
metallic 1.0
```

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"default"` | Material identifier |
| `type` | string | `"principled"` | Material type (currently only `principled`) |
| `baseColor` | float3 | `0.5 0.5 0.5` | Base diffuse/specular color |
| `metallic` | float | `0.0` | 0=dielectric, 1=conductor |
| `roughness` | float | `0.5` | Surface roughness (squared → α for GGX) |
| `ior` | float | `1.5` | Index of refraction |
| `transmission` | float | `0.0` | 0=opaque, 1=fully transmissive |
| `subsurface` | float | `0.0` | Blend to subsurfaceColor in the diffuse approximation |
| `subsurfaceColor` | float3 | `1 1 1` | Diffuse color for the subsurface approximation |
| `anisotropy` | float | `0.0` | Anisotropy in [-1,1]; negative values exchange tangent axes (`anisotropic` is an alias) |
| `emission` | float3 | `0 0 0` | Emissive radiance |
| `emissionTemperature` | float | `6500` | Reserved at 6500; other values are rejected (use RGB emission) |

Reflectances, roughness, metallic, transmission, and subsurface weights must be in [0,1]. IOR must be positive. By default GGX is a single-scattering model; high roughness can lose energy to unmodeled multiple scattering. Subsurface is a local diffuse approximation, not a BSSRDF; `subsurfaceMFP` is reserved. Optional `multipleScattering`, `diffuseRoughness` and opaque `clearcoat` extensions are documented with their supported domains, reciprocal sampling and physical limitations in [MATERIAL_ENERGY.md](MATERIAL_ENERGY.md). Version-2 textures, alpha masks/fractional coverage, normal/bump maps and `absorption` use [SCENE_CONTRACT.md](SCENE_CONTRACT.md).

Emission is one-sided radiance by default. Set `emissionTwoSided 1` to emit the same radiance from both faces (the default is `0`); this matches Cycles mesh-emission sidedness. Two-sided emitters use an absolute emitter cosine in NEE and MIS, and twice the total-power proxy for light selection. Emissive mesh triangles are registered automatically for light sampling. A scene `light mesh_id triangle_id r g b area` entry overrides that triangle's material emission; its legacy area field must be positive, but sampling uses the actual transformed triangle area. Analytic emitters contribute when hit by camera or BSDF rays; they are not sampled by NEE.

## Examples

### Glass
```
name "glass"
baseColor 1.0 1.0 1.0
roughness 0.0
ior 1.5
transmission 1.0
```

### Brushed Metal
```
name "brushed_steel"
baseColor 0.8 0.8 0.82
roughness 0.25
metallic 1.0
anisotropic 0.6
```

## Scene File Usage

Reference a `.mat` file from a `.xenon` scene:
```
matfile "glass.mat"
```
Paths are resolved inside the scene directory’s `materials/` subdirectory.
