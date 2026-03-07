#include "render/wavefront.h"
#include "material/bsdf.h"
#include "camera/sampler.h"
#include "math/simd.h"
#include <algorithm>
#include <cmath>

namespace xn {

void WavefrontRenderer::render_frame_tiled(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain) {
    spp_++;
    
    // Core tiled integrator loop
    pool_->parallel_for_2d(width_, height_, 32, [&](int x_start, int y_start, int x_end, int y_end) {
        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                int pixel_idx = y * width_ + x;
                
                uint64_t seed = (uint64_t)pixel_idx + (uint64_t)spp_ * 0xdeadbeef;
                PCGState rng = seed_pcg(seed);

                float u = (float)x + rng.next_float();
                float v = (float)y + rng.next_float();

                Ray ray = camera.get_ray(u / (float)width_, ((float)height_ - v) / (float)height_);
                Vec3 throughput(1.f);
                Vec3 accumulation(0.f);
                float prev_bsdf_pdf_sa = 0.f;
                bool prev_was_delta = true;

                for (int depth = 0; depth < max_bounces_; ++depth) {
                    HitRecord rec;
                    if (!scene.intersect(ray, rec)) {
                      break;
                    }

                    const Material& mat = scene.materials[rec.mat_id];

                    // ── Direct Emission + MIS ────────────────────────────────
                    // Hit emissive?
                    bool hit_light = false;
                    for (const auto& light : scene.lights) {
                        if (light.tri_idx == (uint32_t)rec.prim_id) {
                            float mis_w = 1.0f;

                            // Only count emission if the light is seen from its front/emitting side
                            float cos_l = std::max(0.0f, dot(rec.geo_normal, -ray.dir));

                            if (cos_l > 1e-6f) {
                                if (depth > 0 && !prev_was_delta) {
                                    // light sampling PDF in area measure, including light selection probability
                                    float light_pdf_area =
                                        (1.0f / light.area) * (1.0f / (float)scene.lights.size());

                                    float dist_sq = rec.t * rec.t;

                                    // convert area PDF to solid-angle PDF
                                    float light_pdf_sa = (light_pdf_area * dist_sq) / cos_l;

                                    mis_w = mis_weight_power2(prev_bsdf_pdf_sa, light_pdf_sa);
                                }

                                accumulation += throughput * light.emission * mis_w;
                            }

                            hit_light = true;
                            break;
                        }
                    }

                    if (hit_light) break;

                    // ── NEE ─────────────────────────────────────────────────────────────
                    if (!scene.lights.empty() && !mat.isDelta) {
                        float pick_pdf = 0.0f;
                        const Light& l = scene.sample_light(rng.next_float(), pick_pdf);

                        const auto& mesh = scene.meshes[l.mesh_id];
                        Vec3 p[3];
                        mesh.get_triangle(l.tri_idx, p[0], p[1], p[2]);

                        // Uniform sample on triangle
                        float u1 = rng.next_float();
                        float u2 = rng.next_float();
                        float su1 = std::sqrt(u1);
                        float b0 = 1.0f - su1;
                        float b1 = u2 * su1;
                        float b2 = 1.0f - b0 - b1;

                        Vec3 light_pos = p[0] * b0 + p[1] * b1 + p[2] * b2;
                        Vec3 light_norm = mesh.geo_normal(l.tri_idx);

                        Vec3 shadow_dir = light_pos - rec.pos;
                        float dist_sq = shadow_dir.length_sq();
                        float dist = std::sqrt(dist_sq);

                        if (dist > 1e-6f) {
                            Vec3 shadow_wi = shadow_dir / dist;

                            // Light must face the shading point
                            float cos_l = std::max(0.0f, dot(light_norm, -shadow_wi));

                            // Use geometric normal consistently while debugging
                            Onb onb(rec.geo_normal);
                            Vec3 wo_local = onb.to_local(-ray.dir);
                            Vec3 wi_local = onb.to_local(shadow_wi);

                            // Surface must also face the sampled direction
                            if (cos_l > 1e-6f && wi_local.z > 1e-6f) {
                                Vec3 f = bsdf_eval_for_nee(wo_local, wi_local, mat);
                                float b_pdf = bsdf_pdf_for_nee(wo_local, wi_local, mat); // solid-angle pdf

                                // area pdf = (1 / area) * pick_pdf
                                // convert to solid angle: p_omega = p_area * dist^2 / cos_light
                                float light_pdf_sa = ((1.0f / l.area) * pick_pdf) * (dist_sq / cos_l);

                                float mis_w = mis_weight_power2(light_pdf_sa, b_pdf);
                                float cos_s = wi_local.z;

                                if (mis_w > 0.0f && light_pdf_sa > 1e-12f) {
                                    Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, shadow_wi, kShadowEps);
                                    Ray shadow_ray{origin, shadow_wi, kShadowEps, dist - kShadowEps};

                                    if (!scene.intersects(shadow_ray)) {
                                        accumulation += throughput * f * cos_s * (l.emission / light_pdf_sa) * mis_w;
                                    }
                                }
                            }
                        }
                    }
                    // ── Russian Roulette ─────────────────────────────────────
                    if (depth >= min_bounces_) {
                        float p = std::max(0.05f, max_component(throughput));
                        if (rng.next_float() > p) break;
                        throughput /= p;
                    }

                    // ── BSDF Sampling ────────────────────────────────────────
                    Onb onb(rec.normal);
                    Vec3 wo_local = onb.to_local(-ray.dir);
                    BSDFSample samp;
                    if (!bsdf_sample(wo_local, mat, rng, samp)) break;

                    if (samp.is_delta) {
                        throughput *= samp.f;
                    } else {
                        float cos_i = std::abs(samp.wi.z);
                        if (samp.pdf < 1e-8f || cos_i < 1e-8f) break;
                        throughput *= samp.f * cos_i / samp.pdf;
                    }

                    Vec3 wi_world = onb.to_world(samp.wi);
                    ray.origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);
                    ray.dir = wi_world;
                    prev_bsdf_pdf_sa = samp.pdf;
                    prev_was_delta = samp.is_delta;
                }
                accumulation_buffer_[pixel_idx] += accumulation;
            }
        }
    });

    // Finalize frame same as wavefront
    float* out = swapchain.get_write_buffer();
    float inv_spp = 1.0f / (float)spp_;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int src_idx = y * width_ + x;
            int dst_idx = (height_ - 1 - y) * width_ + x;
            Vec3 c = accumulation_buffer_[src_idx] * inv_spp;
            out[dst_idx * 3 + 0] = c.x;
            out[dst_idx * 3 + 1] = c.y;
            out[dst_idx * 3 + 2] = c.z;
        }
    }
    swapchain.swap_writer();
}

} // namespace xn
