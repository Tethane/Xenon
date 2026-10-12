#pragma once
// render/swapchain.h — Lock-free triple buffer swapchain for progressive rendering

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <new>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// TripleSwapchain
//   Allows the render thread to write one buffer while the display thread
//   reads another, with a middle buffer to catch the latest completed frame.
// ─────────────────────────────────────────────────────────────────────────────
class TripleSwapchain {
  public:
	TripleSwapchain(int width, int height) : width_(width), height_(height) {
		size_t buf_size = (size_t)width * height * 3 * sizeof(float);

		for (int i = 0; i < 3; ++i) {
			// Align to 64-byte cache lines to avoid false sharing
			buffers_[i] = (float *)aligned_alloc(64, (buf_size + 63) & ~size_t(63));

			if (!buffers_[i]) {
				for (int j = 0; j < i; ++j)
					free(buffers_[j]);
				throw std::bad_alloc();
			}
			std::fill(buffers_[i], buffers_[i] + (size_t)width * height * 3, 0.f);
		}
	}

	~TripleSwapchain() {
		for (int i = 0; i < 3; ++i)
			free(buffers_[i]);
	}

	TripleSwapchain(const TripleSwapchain &) = delete;
	TripleSwapchain &operator=(const TripleSwapchain &) = delete;

	// One producer owns writer_idx_, one consumer owns reader_idx_.
	float *get_write_buffer() {
		return buffers_[writer_idx_];
	}

	void swap_writer() {
		writer_idx_ =
			pending_.exchange(writer_idx_ | kDirty, std::memory_order_acq_rel) & kIndexMask;
	}

	float *get_read_buffer() {
		if (pending_.load(std::memory_order_acquire) & kDirty) {
			reader_idx_ = pending_.exchange(reader_idx_, std::memory_order_acq_rel) & kIndexMask;
		}

		return buffers_[reader_idx_];
	}

	int width() const {
		return width_;
	}

	int height() const {
		return height_;
	}

  private:
	int width_, height_;
	float *buffers_[3];

	static constexpr int kDirty = 4;
	static constexpr int kIndexMask = 3;
	int writer_idx_ = 0;
	int reader_idx_ = 2;
	std::atomic<int> pending_{1};
};

} // namespace xn
