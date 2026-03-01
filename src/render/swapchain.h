#pragma once
// render/swapchain.h — Lock-free triple buffer swapchain for progressive rendering

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// TripleSwapchain
//   Allows the render thread to write one buffer while the display thread
//   reads another, with a middle buffer to catch the latest completed frame.
// ─────────────────────────────────────────────────────────────────────────────
class TripleSwapchain {
public:
    TripleSwapchain(int width, int height)
        : width_(width), height_(height) {
        size_t buf_size = (size_t)width * height * 3 * sizeof(float);
        for (int i = 0; i < 3; ++i) {
            // Align to 64-byte cache lines to avoid false sharing
            buffers_[i] = (float*)aligned_alloc(64, buf_size);
            std::fill(buffers_[i], buffers_[i] + (size_t)width * height * 3, 0.f);
        }
        
        // Indices:
        // writer_idx: buffer currently being rendered into
        // pending_idx: latest finished buffer waiting for display
        // reader_idx: buffer currently being displayed
        writer_idx_.store(0);
        pending_idx_.store(1);
        reader_idx_.store(2);
        
        dirty_.store(false);
    }

    ~TripleSwapchain() {
        for (int i = 0; i < 3; ++i) free(buffers_[i]);
    }

    // Called by Render Thread: Get the buffer to write into
    float* get_write_buffer() {
        return buffers_[writer_idx_.load()];
    }

    // Called by Render Thread: Finished writing, swap writer with pending
    void swap_writer() {
        int old_writer = writer_idx_.load();
        int old_pending = pending_idx_.load();
        
        // Atomically exchange pending with the finished writer
        while (!pending_idx_.compare_exchange_weak(old_pending, old_writer)) {
            // Retry if pending changed (e.g., reader just swapped it)
        }
        
        writer_idx_.store(old_pending);
        dirty_.store(true);
    }

    // Called by Display Thread: Get the buffer to display
    // Returns nullptr if no new frame is available (optional)
    float* get_read_buffer() {
        if (dirty_.load()) {
            int old_reader = reader_idx_.load();
            int old_pending = pending_idx_.load();
            
            // Atomically exchange reader with pending
            while (!pending_idx_.compare_exchange_weak(old_pending, old_reader)) {
                // Retry if pending changed (e.g., writer just swapped it)
            }
            
            reader_idx_.store(old_pending);
            dirty_.store(false);
        }
        return buffers_[reader_idx_.load()];
    }

    int width()  const { return width_; }
    int height() const { return height_; }

private:
    int width_, height_;
    float* buffers_[3];
    
    std::atomic<int> writer_idx_;
    std::atomic<int> pending_idx_;
    std::atomic<int> reader_idx_;
    std::atomic<bool> dirty_;
};

} // namespace xn
