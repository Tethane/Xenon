#include "camera/sampler.h"
#include "camera/camera.h"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace xn;

static int s_pass = 0, s_fail = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL  %s:%d — %s\n", __FILE__, __LINE__, #expr); \
        ++s_fail; \
    } else { ++s_pass; } \
} while(0)
 
#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a)-(b)) < (eps))

void test_pcg() {
    PCGState rng = seed_pcg(42);
    for (int i = 0; i < 100; ++i) {
        float f = rng.next_float();
        CHECK(f >= 0.f && f < 1.f);
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
    cam.look_at({0,0,0}, {0,0,-1}, {0,1,0}, 90.f, 1.f);
    
    // Center ray
    Ray r = cam.get_ray(0.5f, 0.5f);
    CHECK(std::abs(r.dir.x) < 1e-5f);
    CHECK(std::abs(r.dir.y) < 1e-5f);
    CHECK_NEAR(r.dir.z, -1.f, 1e-5f);
}

int main() {
    std::printf("=== sampler_test ===\n");
    test_pcg();
    test_stratified_sampler();
    test_camera();
    std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
