// render/cpu_tiled.cpp — Tiled CPU path integrator (cache-friendly per-tile loop)

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "render/wavefront.h"
#include "material/bsdf.h"
#include "camera/sampler.h"
#include "volume/absorption.h"

namespace xn {

void WavefrontRenderer::render_frame_tiled(const Scene &scene, const Camera &camera,
										   TripleSwapchain &swapchain) {
	spp_++;
	pool_->parallel_for_2d(
		width_, height_, 32, [&](int x_start, int y_start, int x_end, int y_end) {
			for (int y = y_start; y < y_end; ++y) {
				for (int x = x_start; x < x_end; ++x) {
					int pixel_idx = y * width_ + x;
					PCGState rng = seed_path_pcg(uint32_t(pixel_idx), uint32_t(spp_), seed_);
					float u = (float)x + rng.next_float();
					float v = (float)y + rng.next_float();
					Ray ray;

					if (camera.get_lens_radius() > 0.f) {
						float lu = rng.next_float(), lv = rng.next_float();
						ray = camera.get_ray(u / (float)width_, v / (float)height_, lu, lv);
					} else
						ray = camera.get_ray(u / (float)width_, v / (float)height_);
					Vec3 throughput(1.f), accumulation(0.f), prev_pos{};
					float prev_bsdf_pdf_sa = 0.f, eta_scale = 1.f;
					bool prev_was_delta = true;
					AbsorptionStack<Vec3> absorption;

					if (camera.get_medium_material() >= 0)
						absorption.enter(camera.get_medium_material(), camera.get_absorption());

					for (int depth = 0; depth < max_bounces_; ++depth) {
						if (scene.has_alpha_blend)
							ray.alpha_seed = rng.next_uint();
						HitRecord rec;
						bool found = scene.intersect(ray, rec);
						MediumEvent medium =
							sample_medium(scene.media, ray, found ? rec.t : kInfinity, rng);
						throughput *= medium.weight;
						Vec3 interior = absorption.current();

						if (max_component(interior) > 0)
							throughput *= beer_lambert(
								interior, medium.scattered ? (medium.position - ray.origin).length()
														   : (found ? rec.t : kInfinity));

						if (max_component(throughput) == 0)
							break;
						const bool volume = medium.scattered;

						if (!volume && !found) {
							if (scene.sky.active()) {
								float mis = prev_was_delta
												? 1.f
												: mis_weight_power2(prev_bsdf_pdf_sa,
																	scene.sky.pdf(ray.dir));
								accumulation += throughput * scene.sky.evaluate(ray.dir) * mis;
							}
							break;
						}
						Material mat;
						Vec3 position = volume ? medium.position : rec.pos;

						if (!volume) {
							mat = scene.evaluate(rec);
							float weight =
								prev_was_delta
									? 1.f
									: mis_weight_power2(prev_bsdf_pdf_sa,
														scene.light_pdf(rec, prev_pos, ray.dir));
							accumulation += throughput * scene.emission(rec) * weight;
						}
						Vec3 wo = -ray.dir;
						Vec3 ns = volume ? wo : shading_normal(rec, wo, mat.isTransmissive);
						Onb onb(ns);
						Vec3 wo_local = onb.to_local(wo);
						bool last = depth + 1 == max_bounces_;
						bool delta = !volume && mat.isDelta;
						auto evaluate = [&](Vec3 wi, Vec3 &value, float &pdf) {
							if (volume) {
								pdf = henyey_greenstein(dot(ray.dir, wi), medium.g);
								value = Vec3(pdf);

								return true;
							}

							if (!valid_scattering(wo, wi, ns, rec.geo_normal))
								return false;
							Vec3 local = onb.to_local(wi);
							value = bsdf_eval_for_nee(wo_local, local, mat) * std::abs(local.z);
							pdf = bsdf_pdf_for_nee(wo_local, local, mat);

							return true;
						};
						auto origin_for = [&](Vec3 wi) {
							return volume ? position
										  : offset_ray_origin(position, rec.geo_normal, wi,
															  kShadowEps);
						};
						auto visible_contribution = [&](Ray shadow, Vec3 contribution) {
							if (max_component(contribution) <= 0)
								return;

							if (scene.has_alpha_blend)
								shadow.alpha_seed = rng.next_uint();

							if (!scene.intersects(shadow)) {
								Vec3 sigma = absorption.current();

								if (!volume && mat.isTransmissive &&
									dot(wo, rec.geo_normal) * dot(shadow.dir, rec.geo_normal) < 0)
									sigma = rec.front_face ? mat.absorption
														   : absorption.after_leave(rec.mat_id);
								Vec3 attenuation =
									medium_transmittance(scene.media, shadow, shadow.tmax);

								if (max_component(sigma) > 0)
									attenuation *= beer_lambert(sigma, shadow.tmax);
								accumulation += throughput * contribution * attenuation;
							}
						};

						if (!scene.lights.empty() && !delta) {
							float pick_pdf;
							const Light &light = scene.sample_light(
								double(rng.next_uint()) * (1. / 4294967296.), pick_pdf);
							float u1 = rng.next_float(), u2 = rng.next_float(), su = std::sqrt(u1);
							Vec3 light_pos = light.p0 * (1 - su) + light.p1 * (u2 * su) +
											 light.p2 * (su * (1 - u2));
							Vec3 to_light = light_pos - position;
							float dist_sq = to_light.length_sq();

							if (dist_sq > 0) {
								Vec3 wi = to_light / std::sqrt(dist_sq), value;
								float b_pdf;
								float cos_l = dot(light.normal, -wi);

								if (light.emissionTwoSided)
									cos_l = std::abs(cos_l);

								if (cos_l > 0 && evaluate(wi, value, b_pdf)) {
									float pdf = pick_pdf * dist_sq / (cos_l * light.area);
									float mis = last ? 1.f : mis_weight_power2(pdf, b_pdf);
									Vec3 origin = origin_for(wi),
										 target = offset_ray_origin(light_pos, light.normal, -wi,
																	kShadowEps);
									Vec3 segment = target - origin;
									float distance = segment.length();

									if (distance > 0)
										visible_contribution(
											{origin, segment / distance, 0.f, distance},
											value *
												scene.light_emission(light, u2 * su,
																	 su * (1 - u2)) *
												(mis / pdf));
								}
							}
						}

						if (!scene.dir_lights.empty() && !delta) {
							float pick_pdf;
							int idx = scene.sample_dir_light(rng.next_float(), pick_pdf);
							const auto &light = scene.dir_lights[idx];
							Vec3 wi = light.direction, value;
							float pdf;

							if (evaluate(wi, value, pdf))
								visible_contribution({origin_for(wi), wi, 0.f, kInfinity},
													 value * light.emission() / pick_pdf);
						}

						if (!scene.point_lights.empty() && !delta) {
							int idx = std::min(int(rng.next_float() * scene.point_lights.size()),
											   int(scene.point_lights.size()) - 1);
							const auto &light = scene.point_lights[idx];
							Vec3 segment = light.position - position;
							float dist_sq = segment.length_sq();

							if (dist_sq > 0) {
								Vec3 wi = segment / std::sqrt(dist_sq), value;
								float pdf;

								if (evaluate(wi, value, pdf)) {
									Vec3 origin = origin_for(wi);
									segment = light.position - origin;
									float distance = segment.length();

									if (distance > 0)
										visible_contribution(
											{origin, segment / distance, 0.f, distance},
											value * light.intensity *
												(light.falloff(-wi) * scene.point_lights.size() /
												 dist_sq));
								}
							}
						}

						if (scene.sky.active() && !delta) {
							float u1 = rng.next_float(), u2 = rng.next_float();
							auto light = scene.sky.sample(u1, u2);
							Vec3 value;
							float b_pdf;

							if (light.pdf > 0 && evaluate(light.direction, value, b_pdf)) {
								float mis = last ? 1.f : mis_weight_power2(light.pdf, b_pdf);
								visible_contribution(
									{origin_for(light.direction), light.direction, 0.f, kInfinity},
									value * light.radiance * (mis / light.pdf));
							}
						}

						if (last)
							break;
						Vec3 wi;
						float sample_pdf;
						bool sample_delta = false;

						if (volume) {
							float u1 = rng.next_float(), u2 = rng.next_float();
							wi = sample_henyey_greenstein(ray.dir, medium.g, u1, u2);
							sample_pdf = henyey_greenstein(dot(ray.dir, wi), medium.g);
						} else {
							BSDFSample samp;

							if (!bsdf_sample(wo_local, mat, rng, samp))
								break;
							wi = normalize(onb.to_world(samp.wi));

							if (!valid_scattering(wo, wi, ns, rec.geo_normal))
								break;
							throughput *= samp.f * (std::abs(samp.wi.z) / samp.pdf);
							eta_scale *= samp.eta * samp.eta;
							sample_pdf = samp.pdf;
							sample_delta = samp.is_delta;

							if (mat.isTransmissive &&
								dot(wo, rec.geo_normal) * dot(wi, rec.geo_normal) < 0) {
								if (rec.front_face) {
									if ((max_component(mat.absorption) > 0 ||
										 absorption.depth > 0) &&
										!absorption.enter(rec.mat_id, mat.absorption)) {
										accumulation =
											Vec3(std::numeric_limits<float>::quiet_NaN());
										break;
									}
								} else
									absorption.leave(rec.mat_id);
							}
						}

						if (max_component(throughput) == 0)
							break;

						if (depth >= min_bounces_) {
							float p = std::clamp(max_component(throughput) * eta_scale, .05f, .95f);

							if (rng.next_float() >= p)
								break;
							throughput /= p;
						}
						prev_pos = position;
						prev_bsdf_pdf_sa = sample_pdf;
						prev_was_delta = sample_delta;
						ray =
							Ray{volume ? position : offset_ray_origin(position, rec.geo_normal, wi),
								wi};
					}
					accumulation_buffer_[pixel_idx] += accumulation;
				}
			}
		});

	float *out = swapchain.get_write_buffer();
	float inv_spp = 1.f / (float)spp_;

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			Vec3 c = accumulation_buffer_[y * width_ + x] * inv_spp;

			if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z))
				throw std::runtime_error("Non-finite CPU radiance");
			int dst = ((height_ - 1 - y) * width_ + x) * 3; // framebuffer rows run top to bottom
			out[dst] = c.x;
			out[dst + 1] = c.y;
			out[dst + 2] = c.z;
		}
	}
	swapchain.swap_writer();
}

} // namespace xn
