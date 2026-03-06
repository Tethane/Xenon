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
| `name` | string | `"unnamed"` | Material identifier |
| `type` | string | `"principled"` | Material type (currently only `principled`) |
| `baseColor` | float3 | `0.5 0.5 0.5` | Base diffuse/specular color |
| `metallic` | float | `0.0` | 0=dielectric, 1=conductor |
| `roughness` | float | `0.5` | Surface roughness (squared → α for GGX) |
| `ior` | float | `1.5` | Index of refraction |
| `transmission` | float | `0.0` | 0=opaque, 1=fully transmissive |
| `transmissionRoughness` | float | `-1` | Override roughness for BTDF (-1=use roughness) |
| `subsurface` | float | `0.0` | Subsurface scattering weight |
| `subsurfaceColor` | float3 | `0.5 0.5 0.5` | Absorption color for subsurface |
| `anisotropic` | float | `0.0` | Anisotropy in [0,1] |
| `emission` | float3 | `0 0 0` | Emissive radiance |
| `emissionTemperature` | float | `6500` | Color temperature (decorative) |

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
matfile "materials/glass.mat"
```
Paths are resolved relative to the scene file directory.
