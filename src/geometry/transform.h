#pragma once
// geometry/transform.h — Affine TRS transform for TLAS instances.
//
// Convention matches TriangleMesh::transform():
//   v' = scale * Rz(Rx(Ry(v))) + translation
//   i.e. M = Rz * Rx * Ry, local_to_world = s·M + t.
//
// Key property exploited throughout:
//   Ray direction is NOT normalised when converting to local space, so the
//   parametric parameter t is identical in world and local space.  This lets
//   us use world rec.t directly for culling and return it unchanged.

#include <cmath>

#include "math/vec3.h"
#include "math/ray.h"
#include "geometry/aabb.h"

namespace xn {

// ─── Column-major 3×4 affine matrix ──────────────────────────────────────────
//   transform_point(p) = cols[0]*p.x + cols[1]*p.y + cols[2]*p.z + cols[3]
//   i.e. cols[0..2] are the basis vectors, cols[3] is the translation.
struct Mat4x3 {
    Vec3 cols[4];

    static Mat4x3 identity() noexcept {
        Mat4x3 m{};
        m.cols[0]={1,0,0}; m.cols[1]={0,1,0};
        m.cols[2]={0,0,1}; m.cols[3]={0,0,0};
        return m;
    }

    // Apply affine transform to a point (adds translation).
    Vec3 transform_point(Vec3 p) const noexcept {
        return cols[0]*p.x + cols[1]*p.y + cols[2]*p.z + cols[3];
    }
    // Apply linear part only (no translation).
    Vec3 transform_dir(Vec3 d) const noexcept {
        return cols[0]*d.x + cols[1]*d.y + cols[2]*d.z;
    }
    // Apply (upper-3×3)ᵀ to n.
    // When called on world_to_local this yields (A⁻ᵀ)·n — the correct
    // covariant normal transform (A = upper-3×3 of local_to_world).
    //
    // Derivation: (A⁻ᵀ)ᵢⱼ = (A⁻¹)ⱼᵢ = wl.cols[i][j]
    //   ⟹ (A⁻ᵀ·n)ᵢ = Σⱼ wl.cols[i][j]·nⱼ = dot(wl.cols[i], n)
    Vec3 transform_normal(Vec3 n) const noexcept {
        return {
            cols[0].x*n.x + cols[0].y*n.y + cols[0].z*n.z,
            cols[1].x*n.x + cols[1].y*n.y + cols[1].z*n.z,
            cols[2].x*n.x + cols[2].y*n.y + cols[2].z*n.z
        };
    }
};

// ─── Affine TRS transform ────────────────────────────────────────────────────
struct AffineTransform {
    Mat4x3 local_to_world;
    Mat4x3 world_to_local;

    AffineTransform() noexcept
        : local_to_world(Mat4x3::identity()), world_to_local(Mat4x3::identity()) {}

    // Build from (translation, uniform scale, Euler rotation in degrees, YXZ order).
    static AffineTransform from_trs(Vec3 translation, float scale, Vec3 rot_deg) noexcept;

    // Transform world-space ray to local space WITHOUT normalising direction.
    // Consequence: t_local == t_world for all parametric distances — no fixup needed.
    Ray to_local(const Ray& wr) const noexcept {
        Ray lr;
        lr.origin = world_to_local.transform_point(wr.origin);
        lr.dir    = world_to_local.transform_dir(wr.dir);   // intentionally unnormalised
        lr.tmin   = wr.tmin;
        lr.tmax   = wr.tmax;
        return lr;
    }

    // Transform a local normal to world space using the covariant (A⁻ᵀ) formula,
    // then renormalise.  Works correctly for non-uniform scale (though from_trs
    // only produces uniform scale, so this simplifies to pure rotation in practice).
    Vec3 normal_to_world(Vec3 n) const noexcept {
        return normalize(world_to_local.transform_normal(n));
    }

    // Compute world-space AABB by transforming all 8 corners of the local AABB.
    // This is always exact for any affine transform.
    AABB world_aabb(const AABB& local) const noexcept {
        const Vec3& lo = local.mn, &hi = local.mx;
        AABB wb;
        for (int i = 0; i < 8; ++i)
            wb.expand(local_to_world.transform_point({
                (i & 1) ? hi.x : lo.x,
                (i & 2) ? hi.y : lo.y,
                (i & 4) ? hi.z : lo.z
            }));
        return wb;
    }
};

} // namespace xn

// ─── Implementation (header-only — single translation unit friendly) ─────────
namespace xn {
inline AffineTransform AffineTransform::from_trs(Vec3 t, float s, Vec3 rot_deg) noexcept {
    constexpr float kPi = 3.14159265358979323846f;
    const Vec3 rad = rot_deg * (kPi / 180.f);
    const float cx = std::cos(rad.x), sx = std::sin(rad.x);
    const float cy = std::cos(rad.y), sy = std::sin(rad.y);
    const float cz = std::cos(rad.z), sz = std::sin(rad.z);

    // Rx * Ry  (computed symbolically)
    const float rxy[3][3] = {
        {  cy,        0.f,        sy      },
        {  sx * sy,   cx,        -sx * cy },
        { -cx * sy,   sx,         cx * cy }
    };
    // M = Rz * (Rx*Ry)
    const float rz[3][3] = {{ cz,-sz,0},{ sz, cz,0},{ 0, 0,1}};
    float m[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                m[r][c] += rz[r][k] * rxy[k][c];

    // local_to_world: cols[j] = s * column j of M = {m[0][j]*s, m[1][j]*s, m[2][j]*s}
    AffineTransform xf;
    xf.local_to_world.cols[0] = { m[0][0]*s, m[1][0]*s, m[2][0]*s };
    xf.local_to_world.cols[1] = { m[0][1]*s, m[1][1]*s, m[2][1]*s };
    xf.local_to_world.cols[2] = { m[0][2]*s, m[1][2]*s, m[2][2]*s };
    xf.local_to_world.cols[3] = t;

    // world_to_local upper-3×3 = (s·M)⁻¹ = M^T/s
    // wl.cols[j] must satisfy: wl.transform_dir(v)[i] = Σⱼ wl.cols[j][i]·vⱼ = (M^T/s · v)ᵢ
    //   ⟹ wl.cols[j][i] = (M^T/s)[i][j] = M[j][i]/s
    //   ⟹ wl.cols[j] = row j of M / s
    const float inv_s = 1.f / s;
    xf.world_to_local.cols[0] = { m[0][0]*inv_s, m[0][1]*inv_s, m[0][2]*inv_s };  // row 0 / s
    xf.world_to_local.cols[1] = { m[1][0]*inv_s, m[1][1]*inv_s, m[1][2]*inv_s };  // row 1 / s
    xf.world_to_local.cols[2] = { m[2][0]*inv_s, m[2][1]*inv_s, m[2][2]*inv_s };  // row 2 / s
    // Translation: -(M^T/s) · t  so that transform_point(t) == origin
    xf.world_to_local.cols[3] = {
        -(m[0][0]*t.x + m[1][0]*t.y + m[2][0]*t.z) * inv_s,
        -(m[0][1]*t.x + m[1][1]*t.y + m[2][1]*t.z) * inv_s,
        -(m[0][2]*t.x + m[1][2]*t.y + m[2][2]*t.z) * inv_s
    };
    return xf;
}
} // namespace xn
