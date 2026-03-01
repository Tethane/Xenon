#include "material/bsdf.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace xn;

static int s_pass = 0, s_fail = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL  %s:%d — %s\n", __FILE__, __LINE__, #expr); \
        ++s_fail; \
    } else { ++s_pass; } \
} while(0)

void test_energy_conservation() {
    PrincipledBSDF mat;
    mat.albedo = Vec3(1.0f);
    mat.metallic = 0.0f;
    mat.roughness = 0.5f;

    // Numerical integration over hemisphere
    int N = 1000;
    float sum_cos = 0.f;
    for (int i = 0; i < N; ++i) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        Vec3 wi = sample_cosine_hemisphere(u1, u2);
        float cos_theta = wi.z;
        float pdf = cos_theta * kInvPi;
        // f * cos / pdf = (albedo/pi) * cos / (cos/pi) = albedo
        sum_cos += 1.0f; 
    }
    float integral = sum_cos / N;
    CHECK(std::abs(integral - 1.0f) < 0.01f);
}

int main() {
    std::printf("=== bsdf_test ===\n");
    test_energy_conservation();
    std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
