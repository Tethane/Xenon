#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "camera/camera.h"
#include "camera/sampler.h"

using namespace xn;

void require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}

int main() {
	Camera camera;
	camera.look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, 60, 1.5f);
	auto pinhole = camera.get_ray(.7f, .3f);
	camera.set_lens(0, 10);
	auto zero = camera.get_ray(.7f, .3f, .2f, .9f);
	require((zero.origin - pinhole.origin).length() < 1e-7f &&
				(zero.dir - pinhole.dir).length() < 1e-7f,
			"Zero aperture changed pinhole camera");
	camera.set_lens(.2f, 10);
	Vec3 target = pinhole.at(-10 / pinhole.dir.z);
	PCGState rng = seed_pcg(42);
	Vec3 mean{};
	double radius_squared = 0;

	for (int i = 0; i < 20000; ++i) {
		float u = rng.next_float(), v = rng.next_float();
		auto ray = camera.get_ray(.7f, .3f, u, v);
		require(ray.origin.length() <= .100001f, "Lens sample outside aperture");
		require(std::abs(ray.origin.z) < 1e-7f, "Lens origin off lens plane");
		require((ray.at(-10 / ray.dir.z) - target).length() < 1e-4f,
				"Lens rays miss focal plane target");
		mean += ray.origin;
		radius_squared += ray.origin.length_sq();
	}
	require((mean / 20000.f).length() < .001f, "Lens samples are not centered");
	require(std::abs(radius_squared / 20000 - .005) < .0001,
			"Lens samples are not uniform over disk area");
	camera.set_aspect(2);
	require(std::abs(camera.get_horizontal().length() / camera.get_vertical().length() - 2) < 1e-6f,
			"Aspect override failed");
	bool failed = false;

	try {
		camera.set_lens(-1, 3);
	} catch (const std::runtime_error &) {
		failed = true;
	}
	require(failed, "Negative aperture accepted");
	failed = false;

	try {
		camera.set_lens(.1f, 0);
	} catch (const std::runtime_error &) {
		failed = true;
	}
	require(failed, "Nonpositive focal distance accepted");
	std::puts("Camera focal plane and aperture sampling checks passed");
}
