#include "render/swapchain.h"
#include <cstdio>
#include <thread>

using namespace xn;

int main() {
	TripleSwapchain swapchain(7, 5);
	constexpr int size = 7 * 5 * 3;
	constexpr int frames = 20000;
	std::atomic<bool> done(false);
	std::thread producer([&]() {
		for (int frame = 1; frame <= frames; ++frame) {
			float *out = swapchain.get_write_buffer();
			std::fill(out, out + size, static_cast<float>(frame));
			swapchain.swap_writer();
		}
		done.store(true, std::memory_order_release);
	});
	bool valid = true;
	float previous = 0.f;

	while (!done.load(std::memory_order_acquire)) {
		const float *in = swapchain.get_read_buffer();
		float frame = in[0];
		valid &= frame >= previous;

		for (int i = 1; i < size; ++i)
			valid &= in[i] == frame;
		previous = frame;
	}
	producer.join();
	const float *final = swapchain.get_read_buffer();

	for (int i = 0; i < size; ++i)
		valid &= final[i] == frames;
	valid &= swapchain.get_read_buffer() == final;

	if (!valid)
		std::fprintf(stderr, "Torn, stale, or missing final frame\n");

	return valid ? 0 : 1;
}
