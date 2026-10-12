#pragma once
// Scene-linear environment radiance. Lat-long maps use +Y up, longitude
// atan2(z,x), with u=0.5 at +X; rotation shifts texture longitude positively.
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include "math/ray.h"
#include "material/texture.h"

namespace xn {
struct EnvironmentSample {
	Vec3 direction{}, radiance{};
	float pdf = 0;
};

struct Environment {
	Vec3 zenith_color = {.10f, .30f, .80f}, horizon_color = {.55f, .70f, .85f},
		 ground_color = {.10f, .08f, .05f};
	float horizon_sharpness = 3.f, intensity = 1.f;
	std::shared_ptr<const Texture> image;
	std::vector<double> cdf;
	float rotation = 0; // radians

	void set_image(std::shared_ptr<const Texture> texture, float radians = 0) {
		if (!texture || texture->levels.empty())
			throw std::runtime_error("Empty environment image");
		image = std::move(texture);
		rotation = radians;
		int width = image->levels[0].width, height = image->levels[0].height;
		std::vector<double> weights(size_t(width) * height);
		double total = 0;

		for (int y = 0; y < height; ++y) {
			double area =
				(std::cos(double(kPi) * y / height) - std::cos(double(kPi) * (y + 1) / height)) *
				2 * double(kPi) / width;

			for (int x = 0; x < width; ++x) {
				auto p = image->pixels[y * width + x];
				weights[y * width + x] = (0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b) * area;
				total += weights[y * width + x];
			}
		}
		cdf.resize(weights.size() + 1);
		cdf[0] = 0;
		double cumulative = 0;

		for (int y = 0; y < height; ++y) {
			double area =
				(std::cos(double(kPi) * y / height) - std::cos(double(kPi) * (y + 1) / height)) *
				2 * double(kPi) / width;

			for (int x = 0; x < width; ++x) {
				size_t i = size_t(y) * width + x;
				// Uniform support floor prevents zero-probability holes where
				// bilinear interpolation reaches neighboring luminous texels.
				cumulative += (total > 0 ? .95 * weights[i] / total + .05 * area / (4 * double(kPi))
										 : area / (4 * double(kPi)));
				cdf[i + 1] = cumulative;
			}
		}
		cdf.back() = 1.f;
	}

	Vec3 evaluate(Vec3 d) const noexcept {
		if (image) {
			int width = image->levels[0].width, height = image->levels[0].height;
			float u = (std::atan2(d.z, d.x) + rotation) / (2 * kPi) + .5f;
			u -= std::floor(u);
			float v = std::atan2(std::hypot(d.x, d.z), d.y) / kPi;
			float px = u * width - .5f, py = v * height - .5f;
			int x = int(std::floor(px)), y = int(std::floor(py));
			float fx = px - x, fy = py - y;
			auto pixel = [&](int xx, int yy) {
				xx = (xx % width + width) % width;
				yy = std::clamp(yy, 0, height - 1);

				return image->pixels[yy * width + xx].rgb();
			};

			return intensity * lerp(lerp(pixel(x, y), pixel(x + 1, y), fx),
									lerp(pixel(x, y + 1), pixel(x + 1, y + 1), fx), fy);
		}
		float y = d.y;

		if (y < 0)
			return intensity * lerp(horizon_color, ground_color, std::min(-y, 1.f));

		return intensity * lerp(horizon_color, zenith_color, std::pow(y, 1.f / horizon_sharpness));
	}

	float pdf(Vec3 d) const {
		if (!image)
			return 1.f / (4 * kPi);
		int width = image->levels[0].width, height = image->levels[0].height;
		float u = (std::atan2(d.z, d.x) + rotation) / (2 * kPi) + .5f;
		u -= std::floor(u);
		int x = std::min(width - 1, int(u * width));
		int y = std::min(height - 1, int(std::atan2(std::hypot(d.x, d.z), d.y) / kPi * height));
		double area =
			(std::cos(double(kPi) * y / height) - std::cos(double(kPi) * (y + 1) / height)) * 2 *
			double(kPi) / width;
		size_t i = size_t(y) * width + x;

		return area > 0 ? (cdf[i + 1] - cdf[i]) / area : 0;
	}

	EnvironmentSample sample(float u, float v) const {
		EnvironmentSample result;

		if (image) {
			int width = image->levels[0].width, height = image->levels[0].height;
			auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
			int i = std::clamp(int(it - cdf.begin()) - 1, 0, width * height - 1);
			int x = i % width, y = i / width;
			double probability = cdf[i + 1] - cdf[i];
			float residual =
				probability > 0 ? std::clamp((u - cdf[i]) / probability, 0.0, 1.0) : .5f;
			float phi = ((x + residual) / width - .5f) * 2 * kPi - rotation;
			double cos0 = std::cos(double(kPi) * y / height),
				   cos1 = std::cos(double(kPi) * (y + 1) / height);
			double cosine = cos0 + (cos1 - cos0) * v;
			float sine = float(std::sqrt(std::max(0.0, 1 - cosine * cosine)));
			result.direction = {sine * std::cos(phi), float(cosine), sine * std::sin(phi)};
			double area = (cos0 - cos1) * 2 * double(kPi) / width;
			result.pdf = area > 0 ? probability / area : 0;
		} else {
			float cosine = 1 - 2 * u, sine = std::sqrt(std::max(0.f, 1 - cosine * cosine)),
				  phi = 2 * kPi * v;
			result.direction = {sine * std::cos(phi), cosine, sine * std::sin(phi)};
			result.pdf = 1.f / (4 * kPi);
		}
		result.radiance = evaluate(result.direction);

		return result;
	}

	bool active() const noexcept {
		return intensity > 0;
	}
};
} // namespace xn
