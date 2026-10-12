#include "camera/sampler.h"
#include "camera/camera.h"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace xn;

static int s_pass = 0, s_fail = 0;
#define CHECK(expr)                                                                                \
	do {                                                                                           \
		if (!(expr)) {                                                                             \
			std::fprintf(stderr, "  FAIL  %s:%d — %s\n", __FILE__, __LINE__, #expr);               \
			++s_fail;                                                                              \
		} else {                                                                                   \
			++s_pass;                                                                              \
		}                                                                                          \
	} while (0)

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) < (eps))

void test_pcg() {
	PCGState rng = seed_pcg(42);

	for (int i = 0; i < 100; ++i) {
		float f = rng.next_float();
		CHECK(f >= 0.f && f < 1.f);
	}
}

void test_path_streams() {
	auto same = [](PCGState a, PCGState b) { return a.state == b.state && a.inc == b.inc; };
	CHECK(same(seed_path_pcg(17, 23, 42), seed_path_pcg(17, 23, 42)));
	CHECK(!same(seed_path_pcg(17, 23, 0), seed_path_pcg(17, 23, 1ULL << 63)));
	CHECK(!same(seed_path_pcg(0, 1, 42), seed_path_pcg(0xdeadbeefu, 0, 42)));
	CHECK(!same(seed_path_pcg(0xffffffffu, 0, 42), seed_path_pcg(0, 0xffffffffu, 42)));
	// A single long PCG stream passes marginal tests even when path restarts
	// have severe spatial structure. Test integrated independent path fields
	// instead: these eight draws cover camera, lens and first medium variates.
	constexpr int width = 64, height = 48, samples = 1024, dimensions = 8;
	std::vector<double> means(width * height * dimensions, 0);

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			const int pixel = (y + 288) * 512 + x + 64;

			for (int sample = 1; sample <= samples; ++sample) {
				auto rng = seed_path_pcg(pixel, sample, 42);

				for (int d = 0; d < dimensions; ++d)
					means[d * width * height + y * width + x] += rng.next_float() / samples;
			}
		}

	for (int d = 0; d < dimensions; ++d) {
		double mean = 0;

		for (int i = 0; i < width * height; ++i)
			mean += means[d * width * height + i] / (width * height);
		double variance = 0, covariance_x = 0, covariance_y = 0, row_variance = 0;
		auto value = [&](int x, int y) { return means[d * width * height + y * width + x] - mean; };

		for (int y = 0; y < height; ++y) {
			double row = 0;

			for (int x = 0; x < width; ++x) {
				double v = value(x, y);
				variance += v * v;
				row += v / width;

				if (x + 1 < width)
					covariance_x += v * value(x + 1, y);

				if (y + 1 < height)
					covariance_y += v * value(x, y + 1);
			}
			row_variance += row * row / height;
		}
		variance /= width * height;
		double correlation_x = covariance_x / (height * (width - 1) * variance);
		double correlation_y = covariance_y / ((height - 1) * width * variance);
		// Broad statistical bounds, independent of any particular hash output.
		CHECK(std::abs(mean - .5) < .002);
		CHECK(variance * 12 * samples > .75 && variance * 12 * samples < 1.25);
		CHECK(std::abs(correlation_x) < .12);
		CHECK(std::abs(correlation_y) < .12);
		CHECK(row_variance * width / variance < 2.5);
	}
}

void test_stratified_sampler() {
	int nx = 4, ny = 4;
	StratifiedSampler sampler(nx, ny, 123);
	std::vector<float> u, v;
	sampler.get_samples_2d(u, v);

	CHECK(u.size() == 16);
	// Rough check for distribution: each bin should have one sample before shuffle
	// Since we shuffle, we just check they stay in [0,1]
	for (int i = 0; i < 16; ++i) {
		CHECK(u[i] >= 0.f && u[i] <= 1.f);
		CHECK(v[i] >= 0.f && v[i] <= 1.f);
	}
}

void test_camera() {
	Camera cam;
	cam.look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, 90.f, 1.f);

	// Center ray
	Ray r = cam.get_ray(0.5f, 0.5f);
	CHECK(std::abs(r.dir.x) < 1e-5f);
	CHECK(std::abs(r.dir.y) < 1e-5f);
	CHECK_NEAR(r.dir.z, -1.f, 1e-5f);
}

int main() {
	std::printf("=== sampler_test ===\n");
	test_pcg();
	test_path_streams();
	test_stratified_sampler();
	test_camera();
	std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);

	return s_fail ? 1 : 0;
}
