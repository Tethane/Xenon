#pragma once
// Bounded homogeneous RGB media. Coefficients are inverse scene units.
// Boxes are explicit invisible medium boundaries; overlapping boxes are rejected.
#include <algorithm>
#include <cmath>
#include <vector>
#include "camera/sampler.h"
#include "math/ray.h"

namespace xn {
struct HomogeneousMedium {
	Vec3 minimum{}, maximum{};
	Vec3 sigma_a{}, sigma_s{};
	float g = 0;

	Vec3 extinction() const {
		return sigma_a + sigma_s;
	}
};

struct MediumEvent {
	Vec3 weight{1.f};
	Vec3 position{};
	float g = 0;
	bool scattered = false;
};

inline bool medium_segment(const HomogeneousMedium &medium, const Ray &ray, float distance,
						   float &begin, float &end) {
	begin = 0;
	end = distance;

	for (int axis = 0; axis < 3; ++axis) {
		if (std::abs(ray.dir[axis]) < 1e-30f) {
			if (ray.origin[axis] < medium.minimum[axis] || ray.origin[axis] > medium.maximum[axis])
				return false;
		} else {
			float a = (medium.minimum[axis] - ray.origin[axis]) / ray.dir[axis];
			float b = (medium.maximum[axis] - ray.origin[axis]) / ray.dir[axis];

			if (a > b)
				std::swap(a, b);
			begin = std::max(begin, a);
			end = std::min(end, b);

			if (begin >= end)
				return false;
		}
	}

	return begin < end;
}

inline Vec3 beer_lambert(Vec3 sigma, float distance) {
	return {sigma.x > 0 ? std::exp(-sigma.x * distance) : 1.f,
			sigma.y > 0 ? std::exp(-sigma.y * distance) : 1.f,
			sigma.z > 0 ? std::exp(-sigma.z * distance) : 1.f};
}

inline Vec3 medium_transmittance(const std::vector<HomogeneousMedium> &media, const Ray &ray,
								 float distance) {
	Vec3 result(1.f);

	for (const auto &medium : media) {
		float begin, end;

		if (medium_segment(medium, ray, distance, begin, end))
			result *= beer_lambert(medium.extinction(), end - begin);
	}

	return result;
}

inline MediumEvent sample_medium(const std::vector<HomogeneousMedium> &media, const Ray &ray,
								 float distance, PCGState &rng) {
	MediumEvent event;
	float cursor = 0;
	// Disjoint volumes are visited in distance order, independent of declaration.
	for (size_t step = 0; step < media.size(); ++step) {
		int index = -1;
		float nearest = kInfinity, exit = 0;

		for (size_t i = 0; i < media.size(); ++i) {
			float begin, end;

			if (medium_segment(media[i], ray, distance, begin, end) && end > cursor &&
				begin >= cursor && begin < nearest) {
				nearest = begin;
				exit = end;
				index = int(i);
			}
		}

		if (index < 0)
			break;
		const auto &medium = media[index];
		Vec3 sigma = medium.extinction();

		if (max_component(medium.sigma_s) == 0) {
			event.weight *= beer_lambert(sigma, exit - nearest);
			cursor = exit;
			continue;
		}
		int channel = std::min(2, int(rng.next_float() * 3));
		float u = rng.next_float();
		float free_flight = sigma[channel] > 0 ? -std::log1p(-u) / sigma[channel] : kInfinity;
		bool scattered = free_flight < exit - nearest;
		float length = std::min(free_flight, exit - nearest);
		Vec3 transmittance = beer_lambert(sigma, length);
		Vec3 density = scattered ? sigma * transmittance : transmittance;
		float pdf = (density.x + density.y + density.z) / 3.f;

		if (pdf == 0) {
			event.weight = Vec3(0);

			return event;
		}
		event.weight *= transmittance / pdf;

		if (scattered) {
			event.weight *= medium.sigma_s;
			event.scattered = true;
			event.position = ray.at(nearest + free_flight);
			event.g = medium.g;

			return event;
		}
		cursor = exit;
	}

	return event;
}

// Directions here are incoming propagation and outgoing propagation. Thus
// positive g concentrates samples around the incoming propagation direction.
inline float henyey_greenstein(float cosine, float g) {
	float denominator = 1 + g * g - 2 * g * cosine;

	return (1 - g * g) / (4 * kPi * denominator * std::sqrt(denominator));
}

inline Vec3 sample_henyey_greenstein(Vec3 forward, float g, float u, float v) {
	float cosine;

	if (g == 0)
		cosine = 1 - 2 * u;
	else {
		double gd = g;
		double ratio = (1 - gd * gd) / (1 - gd + 2 * gd * u);
		cosine = float((1 + gd * gd - ratio * ratio) / (2 * gd));
	}
	cosine = std::clamp(cosine, -1.f, 1.f);
	float sine = std::sqrt(std::max(0.f, 1 - cosine * cosine));
	float phi = 2 * kPi * v;
	Vec3 tangent = normalize(std::abs(forward.z) < .999f ? cross(Vec3(0, 0, 1), forward)
														 : cross(Vec3(0, 1, 0), forward));
	Vec3 bitangent = cross(forward, tangent);

	return tangent * (sine * std::cos(phi)) + bitangent * (sine * std::sin(phi)) + forward * cosine;
}
} // namespace xn
