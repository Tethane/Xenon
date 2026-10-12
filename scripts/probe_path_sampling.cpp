// Reproduce path-mean fields from docs/SAMPLING.md; output is nine 96x384
// row-major float32 fields (eight variates, then variate5 < .25).
// c++ -O3 -std=c++20 -fopenmp -Isrc scripts/probe_path_sampling.cpp -o /tmp/probe
#include "camera/sampler.h"
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
	if (argc != 4) {
		std::fprintf(stderr,
					 "Usage: probe_path_sampling <0 legacy|1 mixed> <samples> <output.bin>\n");

		return 2;
	}
	int mode = std::atoi(argv[1]), samples = std::atoi(argv[2]);

	if (samples < 1 || (mode != 0 && mode != 1))
		return 2;
	constexpr int width = 384, height = 96, dimensions = 9;
	constexpr uint64_t user_seed = 42;
	std::vector<float> output(width * height * dimensions);
#pragma omp parallel for
	for (int pixel_offset = 0; pixel_offset < width * height; ++pixel_offset) {
		std::array<double, dimensions> sum{};
		uint32_t pixel = (pixel_offset / width + 288) * 512 + pixel_offset % width + 64;

		for (int sample = 1; sample <= samples; ++sample) {
			auto rng = mode ? xn::seed_path_pcg(pixel, sample, user_seed)
							: xn::seed_pcg(pixel + uint64_t(sample) * 0xdeadbeef, user_seed);
			float values[8];

			for (int dimension = 0; dimension < 8; ++dimension) {
				values[dimension] = rng.next_float();
				sum[dimension] += values[dimension];
			}
			sum[8] += values[5] < .25f;
		}

		for (int dimension = 0; dimension < dimensions; ++dimension)
			output[dimension * width * height + pixel_offset] = sum[dimension] / samples;
	}
	FILE *file = std::fopen(argv[3], "wb");

	if (!file)
		return 2;
	bool success = std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size();
	success = std::fclose(file) == 0 && success;

	return success ? 0 : 1;
}
