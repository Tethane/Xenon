#pragma once
#include "cuda/gpu_types.cuh"

namespace xn {
struct GpuMedium {
	float3 minimum{}, maximum{}, sigma_a{}, sigma_s{};
	float g = 0;
};

struct GpuMediumEvent {
	float3 weight = make_f3(1.f), position{};
	float g = 0;
	bool scattered = false;
};

__device__ inline bool gpu_medium_segment(const GpuMedium &medium, const GpuRay &ray,
										  float distance, float &begin, float &end) {
	begin = 0;
	end = distance;

	for (int axis = 0; axis < 3; ++axis) {
		float origin = (&ray.origin.x)[axis], direction = (&ray.dir.x)[axis];
		float lo = (&medium.minimum.x)[axis], hi = (&medium.maximum.x)[axis];

		if (fabsf(direction) < 1e-30f) {
			if (origin < lo || origin > hi)
				return false;
		} else {
			float a = (lo - origin) / direction, b = (hi - origin) / direction;

			if (a > b) {
				float tmp = a;
				a = b;
				b = tmp;
			}
			begin = fmaxf(begin, a);
			end = fminf(end, b);

			if (begin >= end)
				return false;
		}
	}

	return begin < end;
}

__device__ inline float3 gpu_beer_lambert(float3 sigma, float distance) {
	return make_f3(sigma.x > 0 ? expf(-sigma.x * distance) : 1.f,
				   sigma.y > 0 ? expf(-sigma.y * distance) : 1.f,
				   sigma.z > 0 ? expf(-sigma.z * distance) : 1.f);
}

__device__ inline float3 gpu_medium_transmittance(const GpuMedium *media, int count,
												  const GpuRay &ray, float distance) {
	float3 result = make_f3(1.f);

	for (int i = 0; i < count; ++i) {
		float begin, end;

		if (gpu_medium_segment(media[i], ray, distance, begin, end))
			result *= gpu_beer_lambert(media[i].sigma_a + media[i].sigma_s, end - begin);
	}

	return result;
}

__device__ inline GpuMediumEvent gpu_sample_medium(const GpuMedium *media, int count,
												   const GpuRay &ray, float distance, GpuPCG &rng) {
	GpuMediumEvent event;
	float cursor = 0;

	for (int step = 0; step < count; ++step) {
		int index = -1;
		float nearest = kGpuInfinity, exit = 0;

		for (int i = 0; i < count; ++i) {
			float begin, end;

			if (gpu_medium_segment(media[i], ray, distance, begin, end) && end > cursor &&
				begin >= cursor && begin < nearest) {
				nearest = begin;
				exit = end;
				index = i;
			}
		}

		if (index < 0)
			break;
		const auto &medium = media[index];
		float3 sigma = medium.sigma_a + medium.sigma_s;

		if (max_component3(medium.sigma_s) == 0) {
			event.weight *= gpu_beer_lambert(sigma, exit - nearest);
			cursor = exit;
			continue;
		}
		int channel = int(rng.next_float() * 3);
		channel = channel > 2 ? 2 : channel;
		float u = rng.next_float(), coefficient = (&sigma.x)[channel];
		float free_flight = coefficient > 0 ? -log1pf(-u) / coefficient : kGpuInfinity;
		bool scattered = free_flight < exit - nearest;
		float length = fminf(free_flight, exit - nearest);
		float3 transmittance = gpu_beer_lambert(sigma, length);
		float3 density = scattered ? sigma * transmittance : transmittance;
		float pdf = (density.x + density.y + density.z) / 3.f;

		if (pdf == 0) {
			event.weight = make_f3(0);

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

__device__ inline float gpu_henyey_greenstein(float cosine, float g) {
	float denominator = 1 + g * g - 2 * g * cosine;

	return (1 - g * g) / (4 * kGpuPi * denominator * sqrtf(denominator));
}

__device__ inline float3 gpu_sample_henyey_greenstein(float3 forward, float g, float u, float v) {
	float cosine;

	if (g == 0)
		cosine = 1 - 2 * u;
	else {
		double gd = g, ratio = (1 - gd * gd) / (1 - gd + 2 * gd * u);
		cosine = float((1 + gd * gd - ratio * ratio) / (2 * gd));
	}
	cosine = fminf(fmaxf(cosine, -1.f), 1.f);
	float sine = sqrtf(fmaxf(0.f, 1 - cosine * cosine)), phi = 2 * kGpuPi * v;
	float3 tangent = normalize3(fabsf(forward.z) < .999f ? cross3(make_f3(0, 0, 1), forward)
														 : cross3(make_f3(0, 1, 0), forward));
	float3 bitangent = cross3(forward, tangent);

	return tangent * (sine * cosf(phi)) + bitangent * (sine * sinf(phi)) + forward * cosine;
}
} // namespace xn
