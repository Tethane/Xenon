#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "scene/environment.h"
#include "camera/sampler.h"
using namespace xn;

void require(bool b, const char *s) {
	if (!b)
		throw std::runtime_error(s);
}

int main() {
	auto image = std::make_shared<Texture>();
	image->levels = {{16, 8, 0}};
	image->pixels.assign(128, TexturePixel{2, 3, 4, 1});
	Environment env;
	env.set_image(image);
	env.intensity = 1;
	PCGState rng = seed_pcg(92);
	double integral = 0;

	for (int i = 0; i < 100000; ++i) {
		float u = rng.next_float(), v = rng.next_float();
		auto sample = env.sample(u, v);
		require(std::abs(sample.pdf - 1 / (4 * kPi)) < 1e-5,
				"Constant map sampling not uniform solid angle");
		require(std::abs(sample.pdf - env.pdf(sample.direction)) < 1e-5,
				"Environment sample/pdf mismatch");
		require((sample.radiance - Vec3(2, 3, 4)).length() < 1e-5, "Constant HDR radiance changed");
		integral += sample.radiance.x / sample.pdf;
	}
	require(std::abs(integral / 100000 - 8 * kPi) < .001,
			"Environment integral normalization wrong");
	// Bright localized texel and dark surroundings exercise importance sampling,
	// polar rows, longitude wrapping, rotation, and bilinear support floor.
	image->pixels.assign(128, TexturePixel{0, 0, 0, 1});
	image->pixels[3 * 16 + 2] = {100, 20, 1, 1};
	env.set_image(image, .3f);
	double normalization = 0;
	int bright = 0;

	for (int i = 0; i < 100000; ++i) {
		float u = rng.next_float(), v = rng.next_float();
		auto sample = env.sample(u, v);
		require(sample.pdf > 0 && std::isfinite(sample.pdf), "Invalid environment PDF");
		float evaluated = env.pdf(sample.direction);
		require(std::abs(sample.pdf - evaluated) / std::max(sample.pdf, evaluated) < .002,
				"Nonuniform environment sample/pdf mismatch");

		if (sample.radiance.x > 10)
			++bright;
		float cosine = 1 - 2 * rng.next_float(), phi = 2 * kPi * rng.next_float();
		Vec3 uniform{std::sqrt(1 - cosine * cosine) * std::cos(phi), cosine,
					 std::sqrt(1 - cosine * cosine) * std::sin(phi)};
		normalization += env.pdf(uniform) * 4 * kPi;
	}
	require(bright > 85000, "Importance sampler misses concentrated emitter");
	require(std::abs(normalization / 100000 - 1) < .04,
			"Nonuniform environment PDF does not integrate to one");
	auto tall = std::make_shared<Texture>();
	tall->levels = {{1, 20000, 0}};
	tall->pixels.assign(20000, TexturePixel{0, 0, 0, 1});
	tall->pixels[0] = {10000, 10000, 10000, 1};
	env.set_image(tall);
	auto polar = env.sample(.5f, .5f);
	require(polar.pdf > 1 && env.pdf(polar.direction) > 1,
			"Bright polar texel lost its solid angle");
	require(std::abs(polar.pdf / env.pdf(polar.direction) - 1) < 1e-4,
			"High-resolution polar sample/PDF mismatch");
	require(polar.radiance.x > 1000, "Polar lookup lost HDR radiance");
	std::puts("HDR environment radiance, importance PDF, rotation and high-resolution polar "
			  "sampling passed");
}
