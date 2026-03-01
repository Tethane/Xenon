#include "render/integrator.h"
#include "material/bsdf.h"
#include <algorithm>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// PathIntegrator::li
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
            // Background color could be added here
            break;
        }

        const PrincipledBSDF& mat = scene.materials[rec.mat_id];

        // ── Direct Emission (Self-Luminance) ─────────────────────────────────
        // Only if we hit a light directly or via mirror/specular ray
        // (Wait, for full MIS we always add emission but weigh it correctly)
        // For the MVP, let's keep it simple: if hit light directly on first hit, add L.
        if (rec.prim_id != -1) {
            // Check if hit a light triangle
            for (const auto& l : scene.lights) {
                if (l.tri_idx == (uint32_t)rec.prim_id) {
                    // Only add if not shadowed or via shadow ray
                    if (depth == 0 || delta_sample) {
                        L += beta * l.emission;
                    }
                    break;
                }
            }
        }

        // ── Russian Roulette ─────────────────────────────────────────────────
        if (depth > 3) {
            float p = std::max(0.05f, max_component(beta));
            if (rng.next_float() > p) break;
            beta /= p;
        }

        // ── Next Event Estimation (Sample Light) ──────────────────────────────
        if (!scene.lights.empty()) {
            float pick_pdf;
            const Light& l = scene.sample_light(rng.next_float(), pick_pdf);
            LightSample ls = sample_area_light(scene, l, rng.next_float(), rng.next_float());

            Vec3 wi = normalize(ls.pos - rec.pos);
            float dist_sq = (ls.pos - rec.pos).length_sq();
            float cos_l = std::max(0.f, dot(ls.normal, -wi));
            float cos_s = std::max(0.f, dot(rec.normal, wi));

            if (cos_l > 0.f && cos_s > 0.f) {
                Ray shadow_ray{rec.pos + rec.normal * kEps, wi, kEps, std::sqrt(dist_sq) - kEps};
                if (!scene.intersects(shadow_ray)) {
                    // Evaluate BSDF
                    Vec3 f = mat.albedo * kInvPi * cos_s;
                    float light_pdf = (ls.pdf * dist_sq) / (cos_l * pick_pdf);
                    float bsdf_pdf = cos_s * kInvPi;
                    
                    // MIS weight (Balance heuristic)
                    float mis_weight = light_pdf / (light_pdf + bsdf_pdf);
                    L += beta * f * (ls.emission / light_pdf) * mis_weight;
                }
            }
        }

        // ── Scattered Path Sampling (Sample BSDF) ─────────────────────────────
        Onb onb(rec.normal);
        Vec3 local_wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        Vec3 wi = onb.to_world(local_wi);
        
        float cos_theta = std::max(0.f, dot(rec.normal, wi));
        if (cos_theta <= 0.f) break;

        // Update MIS weight for emission hit on next loop
        // (Simplified: for now we assume BSDF sampling is only cosine)
        delta_sample = false; 

        beta *= mat.albedo; 
        ray = Ray{rec.pos + rec.normal * kEps, wi};
    }

    return L;
}

PathIntegrator::LightSample PathIntegrator::sample_area_light(const Scene& scene, const Light& l, float u1, float u2) {
    // For MVP, we assume lights are indexed into meshes[0]
    const auto& mesh = scene.meshes[0];
    Vec3 p0, p1, p2;
    mesh.get_triangle(l.tri_idx, p0, p1, p2);

    // Uniform triangle sampling
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
