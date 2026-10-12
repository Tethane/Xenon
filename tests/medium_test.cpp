#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "volume/medium.h"
#include "volume/absorption.h"
using namespace xn;

void require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}

int main() {
	HomogeneousMedium fog{{-1, -1, -1}, {1, 1, 1}, {.2f, .5f, 1.f}, {0, 0, 0}, 0};
	std::vector<HomogeneousMedium> media{fog};
	Ray ray{{0, 0, 3}, {0, 0, -1}, 0, 10};
	Vec3 tr = medium_transmittance(media, ray, 10);
	require(std::abs(tr.x - std::exp(-.4f)) < 1e-6f && std::abs(tr.y - std::exp(-1.f)) < 1e-6f &&
				std::abs(tr.z - std::exp(-2.f)) < 1e-6f,
			"Beer-Lambert attenuation mismatch");
	auto rng = seed_pcg(23);
	auto sample = sample_medium(media, ray, 10, rng);
	require(!sample.scattered && (sample.weight - tr).length() < 1e-6f,
			"Pure absorption path mismatch");
	require((medium_transmittance(media, {{3, 0, 3}, {0, 0, -1}, 0, 10}, 10) - Vec3(1)).length() <
				1e-6f,
			"Parallel outside ray enters volume");
	require(std::abs(medium_transmittance(media, {{0, 0, 0}, {1, 0, 0}, 0, 10}, 10).y -
					 std::exp(-.5f)) < 1e-6f,
			"Inside medium distance wrong");
	// Independently check free-flight probability and RGB weighting against the
	// analytic integral of single-event scattering over a finite segment.
	media[0].sigma_s = {.4f, .3f, .2f};
	Vec3 sigma = media[0].extinction();
	Vec3 expected_no = beer_lambert(sigma, 2);
	Vec3 expected_scatter = media[0].sigma_s / sigma * (Vec3(1) - expected_no);
	Vec3 no_event{}, scattering{};
	constexpr int count = 400000;

	for (int i = 0; i < count; ++i) {
		auto event = sample_medium(media, ray, 10, rng);

		if (event.scattered)
			scattering += event.weight;
		else
			no_event += event.weight;
	}
	require((no_event / float(count) - expected_no).length() < .004f,
			"RGB free-flight survival estimator biased");
	require((scattering / float(count) - expected_scatter).length() < .004f,
			"RGB free-flight scattering estimator biased");

	for (float g : {-.75f, 0.f, .001f, .7f}) {
		double cosine = 0, normalization = 0;

		for (int i = 0; i < count; ++i) {
			float u = rng.next_float(), v = rng.next_float();
			auto direction = sample_henyey_greenstein({0, 0, 1}, g, u, v);
			cosine += direction.z;
			float c = -1.f + 2.f * (i + .5f) / count;
			normalization += henyey_greenstein(c, g) * 4 * kPi / count;
		}
		require(std::abs(cosine / count - g) < .004,
				"Phase function sampled mean cosine disagrees with g");
		require(std::abs(normalization - 1) < 1e-4, "Phase function does not normalize");
	}
	// Two disjoint media in reverse declaration order must still scatter in the
	// nearer box first, and deterministic absorption must compose correctly.
	auto farther = fog;
	farther.minimum.z = -5;
	farther.maximum.z = -3;
	media = {farther, fog};
	auto both = sample_medium(media, ray, 10, rng);
	require(!both.scattered && (both.weight - tr * tr).length() < 1e-6f,
			"Disjoint medium traversal order wrong");
	AbsorptionStack<Vec3, 2> stack;
	require(max_component(stack.current()) == 0, "Vacuum absorption is nonzero");
	require(stack.enter(2, {.1f, .2f, .3f}), "Could not enter medium");
	require(stack.enter(5, {1, 2, 3}), "Could not enter nested medium");
	require(!stack.enter(7, {1, 1, 1}), "Absorption stack overflow ignored");
	stack.leave(5);
	require((stack.current() - Vec3(.1f, .2f, .3f)).length() < 1e-7,
			"Exiting interior did not restore enclosing absorption");
	stack.leave(2);
	require(max_component(stack.current()) == 0, "Exiting final interior did not restore vacuum");
	std::puts(
		"Beer-Lambert, RGB free-flight, HG distribution and absorption boundary checks passed");
}
