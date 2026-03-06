#include "render/integrator.h"
#include "material/bsdf.h"
#include <algorithm>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// PathIntegrator::li (LEGACY — kept for reference, uses Material now)
//   Unbiased path tracing with Next Event Estimation (NEE) and 
//   Multiple Importance Sampling (MIS)
// ─────────────────────────────────────────────────────────────────────────────
Vec3 PathIntegrator::li(const Ray& initial_ray, const Scene& scene, PCGState& rng) const {
    Vec3 L(0.f);     // Accumulated radiance
    Vec3 beta(1.f);  // Throughput (multiplicative)
    Ray ray = initial_ray;
    
    bool delta_sample = false; // For specular paths in MIS

    for (int depth = 0; depth < max_depth_; ++depth) {
        HitRecord rec;
        if (!scene.intersect(ray, rec)) {
            break;
        }

        const Material& mat = scene.materials[rec.mat_id];

        // ── Direct Emission ──────────────────────────────────────────────
        if (rec.prim_id != -1) {
            for (const auto& l : scene.lights) {
                if (l.tri_idx == (uint32_t)rec.prim_id) {
                    if (depth == 0 || delta_sample) {
                        L += beta * l.emission;
                    }
                    break;
                }
            }
        }

        // ── Russian Roulette ─────────────────────────────────────────────
        if (depth > 3) {
            float p = std::max(0.05f, max_component(beta));
            if (rng.next_float() > p) break;
            beta /= p;
        }

        // ── NEE (Simplified — diffuse only for legacy) ───────────────────
        if (!scene.lights.empty() && !mat.isDelta) {
            float pick_pdf;
            const Light& l = scene.sample_light(rng.next_float(), pick_pdf);
            LightSample ls = sample_area_light(scene, l, rng.next_float(), rng.next_float());

            Vec3 wi = normalize(ls.pos - rec.pos);
            float dist_sq = (ls.pos - rec.pos).length_sq();
            float cos_l = std::max(0.f, dot(ls.normal, -wi));
            float cos_s = std::max(0.f, dot(rec.normal, wi));

            if (cos_l > 0.f && cos_s > 0.f) {
                Ray shadow_ray{rec.pos + rec.normal * kRayEps, wi, kRayEps, std::sqrt(dist_sq) - kRayEps};
                if (!scene.intersects(shadow_ray)) {
                    Vec3 f = mat.baseColor * (1.f - mat.metallic) * kInvPi * cos_s;
                    float light_pdf = (ls.pdf * dist_sq) / (cos_l * pick_pdf);
                    float bsdf_pdf_val = cos_s * kInvPi;
                    
                    float mis_weight = light_pdf / (light_pdf + bsdf_pdf_val);
                    L += beta * f * (ls.emission / light_pdf) * mis_weight;
                }
            }
        }

        // ── BSDF Sampling (cosine hemisphere for legacy compatibility) ────
        Onb onb(rec.normal);
        Vec3 local_wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        Vec3 wi = onb.to_world(local_wi);
        
        float cos_theta = std::max(0.f, dot(rec.normal, wi));
        if (cos_theta <= 0.f) break;

        delta_sample = false; 
        beta *= mat.baseColor * (1.f - mat.metallic);
        ray = Ray{rec.pos + rec.normal * kRayEps, wi};
    }

    return L;
}

PathIntegrator::LightSample PathIntegrator::sample_area_light(const Scene& scene, const Light& l, float u1, float u2) {
    const auto& mesh = scene.meshes[0];
    Vec3 p0, p1, p2;
    mesh.get_triangle(l.tri_idx, p0, p1, p2);

    float su1 = std::sqrt(u1);
    float u = 1.0f - su1;
    float v = u2 * su1;
    
    LightSample ls;
    ls.pos = p0 * (1.f - u - v) + p1 * u + p2 * v;
    ls.normal = mesh.geo_normal(l.tri_idx);
    ls.emission = l.emission;
    ls.pdf = 1.f / l.area;
    return ls;
}

} // namespace xn
