#include "render/wavefront.h"
#include "camera/sampler.h"
#include "math/simd.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace xn {

inline float mis_weight_power2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

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
    WavefrontQueue<RayWorkItem> q_rays;
    WavefrontQueue<HitWorkItem> q_hits;
    WavefrontQueue<ShadowWorkItem> q_shadow;
    WavefrontQueue<RayWorkItem> q_rays_next;
    
    // Categorized hit queues
    WavefrontQueue<HitWorkItem> q_hit_diffuse;
    WavefrontQueue<HitWorkItem> q_hit_glossy_refl;
    WavefrontQueue<HitWorkItem> q_hit_glossy_trans;
    WavefrontQueue<HitWorkItem> q_hit_delta;

    q_rays.reset(num_paths);
    q_hits.reset(num_paths);
    q_shadow.reset(num_paths);
    q_rays_next.reset(num_paths);
    
    q_hit_diffuse.reset(num_paths);
    q_hit_glossy_refl.reset(num_paths);
    q_hit_glossy_trans.reset(num_paths);
    q_hit_delta.reset(num_paths);

    // ── RayGenKernel ─────────────────────────────────────────────────────────
    for (int i = 0; i < num_paths; ++i) {
        int x = tile.x0 + (i % tile_w);
        int y = tile.y0 + (i / tile_w);
        uint64_t seed = (uint64_t)y * width_ + x + (uint64_t)spp_ * 0xdeadbeef;
        
        paths[i].rng = seed_pcg(seed);
        float u = (float)x + paths[i].rng.next_float();
        float v_coord = (float)y + paths[i].rng.next_float();
        
        paths[i].ray = camera.get_ray(u / (float)width_, ((float)height_ - v_coord) / (float)height_);
        paths[i].throughput = Vec3(1.f);
        paths[i].radiance   = Vec3(0.f);
        paths[i].pixel_idx  = y * width_ + x;
        paths[i].depth      = 0;
        paths[i].active     = true;
        paths[i].specular   = true;
        paths[i].prev_was_delta = true;
        paths[i].prev_bsdf_pdf_sa = 1.0f;
        q_rays.push({i});
    }

    // ── Wavefront Pipeline ───────────────────────────────────────────────────
    for (int bounce = 0; bounce < max_bounces_; ++bounce) {
        if (q_rays.size == 0) break;

        // 1. IntersectKernel
        q_hits.size = 0;
        for (int i = 0; i < q_rays.size; ++i) {
            int p_idx = q_rays.items[i].path_idx;
            HitRecord rec;
            if (scene.intersect(paths[p_idx].ray, rec)) {
                q_hits.push({p_idx, rec});
            } else {
                paths[p_idx].active = false; // Miss
            }
        }

        // 2. ClassificationKernel
        q_hit_diffuse.size = 0;
        q_hit_glossy_refl.size = 0;
        q_hit_glossy_trans.size = 0;
        q_hit_delta.size = 0;

        for (int i = 0; i < q_hits.size; ++i) {
            const auto& work = q_hits.items[i];
            const PrincipledBSDF& mat = scene.materials[work.hit.mat_id];
            
            // Simple classification logic
            if (mat.roughness < 0.01f || (mat.transmission > 0.0f && mat.transmission_roughness < 0.01f)) {
                q_hit_delta.push(work);
            } else if (mat.transmission > 0.1f) {
                q_hit_glossy_trans.push(work);
            } else if (mat.metallic > 0.1f || mat.roughness < 0.2f) {
                q_hit_glossy_refl.push(work);
            } else {
                q_hit_diffuse.push(work);
            }
        }

        // 3. Shading Kernels (simplified for now to 1-2 kernels to keep it reviewable)
        auto shade_task = [&](WavefrontQueue<HitWorkItem>& q) {
            for (int i = 0; i < q.size; ++i) {
                const auto& work = q.items[i];
                PathState& path = paths[work.path_idx];
                const HitRecord& rec = work.hit;
                const PrincipledBSDF& mat = scene.materials[rec.mat_id];

                // a. Emissive hit (MIS)
                bool hit_light = false;
                for (const auto& light : scene.lights) {
                    if (light.tri_idx == (uint32_t)rec.prim_id) {
                        float mis_w = 1.0f;
                        if (!path.prev_was_delta) {
                            float light_pdf_area = 1.0f / (light.area * (float)scene.lights.size());
                            float dist_sq = rec.t * rec.t;
                            float cos_l = std::abs(dot(rec.geo_normal, -path.ray.dir));
                            if (cos_l > 0) {
                                float light_pdf_sa = (light_pdf_area * dist_sq) / cos_l;
                                mis_w = mis_weight_power2(path.prev_bsdf_pdf_sa, light_pdf_sa);
                            }
                        }
                        path.radiance += path.throughput * light.emission * mis_w;
                        hit_light = true;
                        break;
                    }
                }
                if (hit_light) { path.active = false; continue; }

                // b. NEE
                if (!scene.lights.empty()) {
                    float pick_pdf;
                    const Light& l = scene.sample_light(path.rng.next_float(), pick_pdf);
                    const auto& mesh = scene.meshes[0]; 
                    Vec3 p[3]; mesh.get_triangle(l.tri_idx, p[0], p[1], p[2]);
                    float su1 = std::sqrt(path.rng.next_float());
                    Vec3 light_pos = p[0]*(1.f-su1) + p[1]*(su1*(1.f-path.rng.next_float())) + p[2]*(su1*path.rng.next_float());
                    Vec3 light_norm = mesh.geo_normal(l.tri_idx);

                    Vec3 shadow_wi = normalize(light_pos - rec.pos);
                    float dist_sq = (light_pos - rec.pos).length_sq();
                    float cos_l = std::abs(dot(light_norm, -shadow_wi));
                    
                    Onb onb(rec.normal);
                    Vec3 wo_local = onb.to_local(-path.ray.dir);
                    Vec3 wi_local = onb.to_local(shadow_wi);

                    if (cos_l > 0.001f && wi_local.z > 0.001f) {
                        Vec3 f = bsdf_eval(wo_local, wi_local, mat);
                        float b_pdf = bsdf_pdf(wo_local, wi_local, mat);
                        float light_pdf_sa = (1.f / l.area) * (dist_sq / cos_l) / pick_pdf;
                        float mis_w = mis_weight_power2(light_pdf_sa, b_pdf);
                        
                        Vec3 contrib = path.throughput * f * (l.emission / light_pdf_sa) * mis_w;
                        Ray shadow_ray{rec.pos + rec.normal * kEps, shadow_wi, kEps, std::sqrt(dist_sq) - kEps};
                        q_shadow.push({work.path_idx, shadow_ray, shadow_ray.tmax, contrib});
                    }
                }

                // c. Russian Roulette
                if (path.depth >= min_bounces_) {
                    float p = std::max(0.05f, max_component(path.throughput));
                    if (path.rng.next_float() > p) { path.active = false; continue; }
                    path.throughput /= p;
                }

                // d. Sample BSDF
                Onb onb(rec.normal);
                Vec3 wo_local = onb.to_local(-path.ray.dir);
                BSDFSample sample;
                if (!bsdf_sample(wo_local, mat, path.rng, sample)) { path.active = false; continue; }

                path.throughput *= sample.f;
                path.ray = Ray{rec.pos + rec.normal * (sample.wi.z > 0 ? kEps : -kEps), onb.to_world(sample.wi)};
                path.prev_bsdf_pdf_sa = sample.pdf;
                path.prev_was_delta = sample.is_delta;
                path.depth++;
                q_rays_next.push({work.path_idx});
            }
        };

        shade_task(q_hit_diffuse);
        shade_task(q_hit_glossy_refl);
        shade_task(q_hit_glossy_trans);
        shade_task(q_hit_delta);

        // 4. ShadowKernel
        for (int i = 0; i < q_shadow.size; ++i) {
            const auto& work = q_shadow.items[i];
            if (!scene.intersects(work.ray)) {
                paths[work.path_idx].radiance += work.contrib;
            }
        }

        q_rays.items = q_rays_next.items;
        q_rays.size.store(q_rays_next.size.load());
        q_rays_next.size = 0;
        q_shadow.size = 0;
    }

    // ── Accumulate Kernel ────────────────────────────────────────────────────
    for (int i = 0; i < num_paths; ++i) {
        accumulation_buffer_[paths[i].pixel_idx] += paths[i].radiance;
    }
}

} // namespace xn
