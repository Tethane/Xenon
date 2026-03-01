#pragma once
// camera/sampler.h — Low-discrepancy stratified sampler

#include <random>
#include <vector>
#include <algorithm>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// PCG Hash for fast, high-quality RNG
// ─────────────────────────────────────────────────────────────────────────────
struct PCGState {
    uint64_t state;
    uint64_t inc;

    uint32_t next_uint() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + (inc | 1);
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    float next_float() {
        return (next_uint() >> 8) * (1.f / 16777216.f);
    }
};

inline PCGState seed_pcg(uint64_t seed, uint64_t seq = 0) {
    PCGState s = {0, (seq << 1u) | 1u};
    s.next_uint();
    s.state += seed;
    s.next_uint();
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// StratifiedSampler — produces jittered grid samples
// ─────────────────────────────────────────────────────────────────────────────
class StratifiedSampler {
public:
    StratifiedSampler(int x_samples, int y_samples, uint64_t seed)
        : nx_(x_samples), ny_(y_samples) {
        rng_ = seed_pcg(seed);
    }

    // Generate 2D samples for a pixel (e.g., for AA)
    void get_samples_2d(std::vector<float>& u, std::vector<float>& v) {
        u.resize(nx_ * ny_);
        v.resize(nx_ * ny_);
        float dx = 1.f / nx_;
        float dy = 1.f / ny_;

        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                u[j * nx_ + i] = (i + rng_.next_float()) * dx;
                v[j * nx_ + i] = (j + rng_.next_float()) * dy;
            }
        }

        // Shuffle samples to decorrelate dimensions
        shuffle(u);
        shuffle(v);
    }

    float next_1d() { return rng_.next_float(); }
    void  next_2d(float& x, float& y) { x = rng_.next_float(); y = rng_.next_float(); }

private:
    int nx_, ny_;
    PCGState rng_;

    void shuffle(std::vector<float>& vec) {
        for (int i = (int)vec.size() - 1; i > 0; --i) {
            int j = rng_.next_uint() % (i + 1);
            std::swap(vec[i], vec[j]);
        }
    }
};

} // namespace xn
