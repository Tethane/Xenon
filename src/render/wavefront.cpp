#include "render/wavefront.h"
#include "camera/sampler.h"
#include "math/simd.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace xn {

WavefrontRenderer::WavefrontRenderer(int width, int height)
    : width_(width), height_(height) {
    accumulation_buffer_.resize(width * height, Vec3(0.f));
    pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency());

    // Create 16x16 tiles
    constexpr int kTileSize = 16;
    for (int y = 0; y < height; y += kTileSize) {
        for (int x = 0; x < width; x += kTileSize) {
            tiles_.push_back({x, y, std::min(x + kTileSize, width), std::min(y + kTileSize, height)});
        }
    }
}

WavefrontRenderer::~WavefrontRenderer() = default;

void WavefrontRenderer::render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain) {
    spp_++;
    
    for (const auto& tile : tiles_) {
        pool_->enqueue([this, tile, &scene, &camera] {
            render_tile(tile, scene, camera);
        });
    }
    pool_->wait();

    // Copy to swapchain and normalize
    float* out = swapchain.get_write_buffer();
    float inv_spp = 1.0f / (float)spp_;
    for (int i = 0; i < width_ * height_; ++i) {
        Vec3 c = accumulation_buffer_[i] * inv_spp;
        out[i * 3 + 0] = c.x;
        out[i * 3 + 1] = c.y;
        out[i * 3 + 2] = c.z;
    }
    swapchain.swap_writer();
}

void WavefrontRenderer::render_tile(const Tile& tile, const Scene& scene, const Camera& camera) {
    int tile_w = tile.x1 - tile.x0;
    int tile_h = tile.y1 - tile.y0;
    int num_paths = tile_w * tile_h;

    std::vector<PathState> paths(num_paths);
    WavefrontQueue active_q;
    WavefrontQueue next_q;
    active_q.reset(num_paths);
    next_q.reset(num_paths);

    uint64_t seed = (uint64_t)tile.y0 * width_ + tile.x0 + spp_ * 1234567;
    PCGState rng = seed_pcg(seed);

    // ── Generate Path Kernel ─────────────────────────────────────────────────
    for (int i = 0; i < num_paths; ++i) {
        int lx = i % tile_w;
        int ly = i / tile_w;
        int x = tile.x0 + lx;
        int y = tile.y0 + ly;

        float u = (float)x + rng.next_float();
        float v = (float)y + rng.next_float();
        
        paths[i].ray = camera.get_ray(u / width_, v / height_);
        paths[i].throughput = Vec3(1.f);
        paths[i].radiance   = Vec3(0.f);
        paths[i].pixel_idx  = y * width_ + x;
        paths[i].active     = true;
        paths[i].specular   = true; // Camera rays are "specular" for MIS
        active_q.push(i);
    }

    // ── Bounce Loop ──────────────────────────────────────────────────────────
    for (int bounce = 0; bounce < max_bounces_; ++bounce) {
        if (active_q.size == 0) break;

        // ── Intersect Kernel ─────────────────────────────────────────────────
        std::vector<HitRecord> hits(active_q.size);
        for (int q_idx = 0; q_idx < active_q.size; ++q_idx) {
            int p_idx = active_q.indices[q_idx];
            scene.intersect(paths[p_idx].ray, hits[q_idx]);
        }

        // ── Shade Kernel ─────────────────────────────────────────────────────
        next_q.size = 0;
        for (int q_idx = 0; q_idx < active_q.size; ++q_idx) {
            int p_idx = active_q.indices[q_idx];
            PathState& path = paths[p_idx];
            HitRecord& rec  = hits[q_idx];

            if (!rec.valid()) {
                // Background radiance could be added here
                path.active = false;
                continue;
            }

            const PrincipledBSDF& mat = scene.materials[rec.mat_id];

            // 1. Direct Emission (Hit Light)
            bool hit_light = false;
            for (const auto& light : scene.lights) {
                if (light.tri_idx == (uint32_t)rec.prim_id) {
                    float mis_weight = 1.0f;
                    if (bounce > 0 && !path.specular) {
                        float cos_s = std::abs(dot(rec.normal, -path.ray.dir));
                        float bsdf_pdf = cos_s * kInvPi;
                        float light_pdf = 1.0f / (light.area * (float)scene.lights.size());
                        // Convert light area pdf to solid angle pdf
                        float dist_sq = rec.t * rec.t;
                        float cos_l = std::abs(dot(rec.geo_normal, -path.ray.dir));
                        if (cos_l > 0) {
                            float light_pdf_sa = (light_pdf * dist_sq) / cos_l;
                            mis_weight = bsdf_pdf / (bsdf_pdf + light_pdf_sa);
                        }
                    }
                    path.radiance += path.throughput * light.emission * mis_weight;
                    hit_light = true;
                    break;
                }
            }
            if (hit_light) {
                path.active = false;
                continue;
            }

            // 2. Next Event Estimation (NEE)
            if (!scene.lights.empty()) {
                float pick_pdf;
                const Light& l = scene.sample_light(rng.next_float(), pick_pdf);
                
                const auto& mesh = scene.meshes[0]; // Assuming single mesh for lights in MVP
                Vec3 p0, p1, p2;
                mesh.get_triangle(l.tri_idx, p0, p1, p2);
                float su1 = std::sqrt(rng.next_float());
                float u = 1.0f - su1;
                float v = rng.next_float() * su1;
                Vec3 light_pos = p0 * (1.f - u - v) + p1 * u + p2 * v;
                Vec3 light_norm = mesh.geo_normal(l.tri_idx);

                Vec3 wi = normalize(light_pos - rec.pos);
                float dist_sq = (light_pos - rec.pos).length_sq();
                float cos_l = std::max(0.f, dot(light_norm, -wi));
                float cos_s = std::max(0.f, dot(rec.normal, wi));

                if (cos_l > 0.f && cos_s > 0.f) {
                    Ray shadow_ray{rec.pos + rec.normal * kEps, wi, kEps, std::sqrt(dist_sq) - kEps};
                    if (!scene.intersects(shadow_ray)) {
                        Vec3 f = mat.albedo * kInvPi;
                        float light_pdf = (1.f / l.area) * (dist_sq / cos_l) / pick_pdf;
                        float bsdf_pdf = cos_s * kInvPi;
                        float mis_weight = light_pdf / (light_pdf + bsdf_pdf);
                        path.radiance += path.throughput * f * (l.emission / light_pdf) * cos_s * mis_weight;
                    }
                }
            }

            // 3. Russian Roulette
            if (bounce > min_bounces_) {
                float p = std::max(0.05f, max_component(path.throughput));
                if (rng.next_float() > p) {
                    path.active = false;
                    continue;
                }
                path.throughput /= p;
            }

            // 4. Sample BSDF
            Onb onb(rec.normal);
            float u1 = rng.next_float(), u2 = rng.next_float();
            Vec3 local_wi = sample_cosine_hemisphere(u1, u2);
            Vec3 wi = onb.to_world(local_wi);
            float cos_s = std::max(0.f, dot(rec.normal, wi));
            
            if (cos_s <= 0.f) {
                path.active = false;
                continue;
            }

            path.throughput *= mat.albedo; // cos_s cancels with PDF
            path.ray = Ray{rec.pos + rec.normal * kEps, wi};
            path.specular = false;
            next_q.push(p_idx);
        }

        active_q = next_q;
    }

    // ── Accumulate Kernel ────────────────────────────────────────────────────
    for (int i = 0; i < num_paths; ++i) {
        accumulation_buffer_[paths[i].pixel_idx] += paths[i].radiance;
    }
}

} // namespace xn
