#include "render/wavefront_state.h"
#include <cstdio>
#include <vector>

using namespace xn;

static int s_pass = 0, s_fail = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL  %s:%d — %s\n", __FILE__, __LINE__, #expr); \
        ++s_fail; \
    } else { ++s_pass; } \
} while(0)

void test_queue_logic() {
    WavefrontQueue q;
    q.reset(10);
    CHECK(q.size == 0);

    q.push(5);
    q.push(2);
    CHECK(q.size == 2);
    CHECK(q.indices[0] == 5);
    CHECK(q.indices[1] == 2);

    q.reset(10);
    CHECK(q.size == 0);
}

void test_path_state_init() {
    PathState state;
    CHECK(state.active == false);
    CHECK(state.throughput.x == 1.f);
    CHECK(state.radiance.x == 0.f);
}

int main() {
    std::printf("=== wavefront_test ===\n");
    test_queue_logic();
    test_path_state_init();
    std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
