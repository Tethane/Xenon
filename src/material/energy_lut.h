#pragma once
#include <cmath>

// Shared CPU/device directional-albedo table. Integrates Xenon's existing
// height-correlated Smith GGX (not an unrelated game-engine approximation).
namespace xn::energy {
constexpr int rough_count = 33, cos_count = 65, ior_count = 33;
static const float host_table[] = {
#include "energy_lut_data.inc"
};
#ifdef __CUDACC__
static __device__ const float device_table[] = {
#include "energy_lut_data.inc"
};
#define XN_ENERGY_HD __host__ __device__
#else
#define XN_ENERGY_HD
#endif
XN_ENERGY_HD inline float clamp(float x, float a, float b) {
	return fminf(fmaxf(x, a), b);
}

XN_ENERGY_HD inline float raw(int index) {
#ifdef __CUDA_ARCH__
	return device_table[index];
#else
	return host_table[index];
#endif
}

XN_ENERGY_HD inline float texel(int channel, int rough, int cosine) {
	return raw((channel * rough_count + rough) * cos_count + cosine);
}

XN_ENERGY_HD inline float channel_energy(int channel, float rough, float cosine) {
	float r = clamp(rough, 0, 1) * (rough_count - 1),
		  c = clamp(fabsf(cosine), 0, 1) * (cos_count - 1);
	int r0 = int(r), r1 = r0 + 1 < rough_count ? r0 + 1 : r0, c0 = int(c),
		c1 = c0 + 1 < cos_count ? c0 + 1 : c0;
	float a = texel(channel, r0, c0) * (1 - (c - c0)) + texel(channel, r0, c1) * (c - c0);
	float b = texel(channel, r1, c0) * (1 - (c - c0)) + texel(channel, r1, c1) * (c - c0);

	return a + (b - a) * (r - r0);
}

XN_ENERGY_HD inline float ggx(float rough, float cosine) {
	return channel_energy(0, rough, cosine);
}

XN_ENERGY_HD inline float dielectric(float rough, float cosine, float ior) {
	if (ior == 1.f)
		return 0.f;
	float f = sqrtf(clamp((ior - 1.f) * .5f, 0, 1)) * (ior_count - 1);
	int lo = int(f), hi = lo + 1 < ior_count ? lo + 1 : lo;
	float a = channel_energy(lo + 1, rough, cosine), b = channel_energy(hi + 1, rough, cosine);

	return a + (b - a) * (f - lo);
}

// Integrate each piecewise-linear LUT interval against 2*cos(theta). The exact
// integral of this interpolant ensures the reciprocal compensation normalizes.
XN_ENERGY_HD inline float ggx_average(float rough) {
	float r = clamp(rough, 0, 1) * (rough_count - 1);
	int lo = int(r), hi = lo + 1 < rough_count ? lo + 1 : lo;
	constexpr int offset = rough_count * cos_count * (ior_count + 1);

	return raw(offset + lo) + (raw(offset + hi) - raw(offset + lo)) * (r - lo);
}

XN_ENERGY_HD inline float dielectric_average(float ior) {
	float f = sqrtf(clamp((ior - 1.f) * .5f, 0, 1)) * (ior_count - 1);
	int lo = int(f), hi = lo + 1 < ior_count ? lo + 1 : lo;
	constexpr int offset = rough_count * cos_count * (ior_count + 1) + rough_count;

	return raw(offset + lo) + (raw(offset + hi) - raw(offset + lo)) * (f - lo);
}

XN_ENERGY_HD inline float dielectric_ggx_average(float rough, float ior) {
	float r = clamp(rough, 0, 1) * (rough_count - 1),
		  f = sqrtf(clamp((ior - 1.f) * .5f, 0, 1)) * (ior_count - 1);
	int r0 = int(r), r1 = r0 + 1 < rough_count ? r0 + 1 : r0, i0 = int(f),
		i1 = i0 + 1 < ior_count ? i0 + 1 : i0;
	constexpr int offset = rough_count * cos_count * (ior_count + 1) + rough_count + ior_count;
	float a =
		raw(offset + i0 * rough_count + r0) +
		(raw(offset + i0 * rough_count + r1) - raw(offset + i0 * rough_count + r0)) * (r - r0);
	float b =
		raw(offset + i1 * rough_count + r0) +
		(raw(offset + i1 * rough_count + r1) - raw(offset + i1 * rough_count + r0)) * (r - r0);

	return a + (b - a) * (f - i0);
}

XN_ENERGY_HD inline float multiple_fresnel(float average_fresnel, float average_energy) {
	return average_fresnel * average_fresnel * average_energy /
		   fmaxf(1e-7f, 1 - average_fresnel * (1 - average_energy));
}

// Fujii Oren–Nayar directional albedo, polynomial from the EON paper (2025).
XN_ENERGY_HD inline float fon(float rough, float cosine) {
	float x = 1 - fabsf(cosine);
	float g = x * (.0571085289f + x * (.491881867f + x * (-.332181442f + x * .0714429953f)));

	return (1 + rough * g) / (1 + (.5f - 2.f / (3.f * 3.14159265358979323846f)) * rough);
}

XN_ENERGY_HD inline float fon_average(float rough) {
	constexpr float pi = 3.14159265358979323846f;

	return (1 + (2.f / 3.f - 28.f / (15.f * pi)) * rough) / (1 + (.5f - 2.f / (3.f * pi)) * rough);
}

#undef XN_ENERGY_HD
} // namespace xn::energy
