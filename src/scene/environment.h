#pragma once
// scene/environment.h — Pluggable environment / sky model
//
// Interface contract:
//   evaluate(d)  — return emitted radiance for a ray with direction d.
//                  Called from the integrator miss path.
//   sample()     — (future) importance-sample a direction + pdf.
//
// The gradient Sky implementation below is intentionally cheap.
// Replace it with an HDRI texture lookup by switching evaluate() to
// sample a latitude-longitude image without touching any other code.

#include "math/vec3.h"
#include <cmath>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Environment — abstract gradient sky
//
// Convention: d is a unit direction in world space, y-up.
//   y >  0   : above horizon → lerp(horizon, zenith, t^sharpness)
//   y <= 0   : below horizon → ground_color
// ─────────────────────────────────────────────────────────────────────────────
struct Environment {
    Vec3  zenith_color   = {0.10f, 0.30f, 0.80f}; // top-of-sky
    Vec3  horizon_color  = {0.55f, 0.70f, 0.85f}; // near-horizon
    Vec3  ground_color   = {0.10f, 0.08f, 0.05f}; // below the horizon
    float horizon_sharpness = 3.0f;                // exponent (>1 = faster falloff)
    float intensity = 1.0f;                        // overall scale

    // Returns emitted radiance for a ray with unit direction `d`.
    Vec3 evaluate(Vec3 d) const noexcept {
        float y = d.y; // vertical component, y-up world

        if (y < 0.f) {
            // Below horizon — blend to ground color
            float t = std::min(-y, 1.f); // 0 at horizon, 1 straight down
            return intensity * lerp(horizon_color, ground_color, t);
        } else {
            // Above horizon — zenith gradient
            float t = std::pow(y, 1.0f / horizon_sharpness); // nonlinear: slow near horizon
            return intensity * lerp(horizon_color, zenith_color, t);
        }
    }

    // Returns true if this environment contributes any light
    // (i.e., it is not completely black).
    bool active() const noexcept { return intensity > 0.f; }
};

} // namespace xn
