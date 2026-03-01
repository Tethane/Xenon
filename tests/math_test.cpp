// tests/math_test.cpp — unit tests for math library
// Simple, self-contained test runner (no external framework)

#include "math/vec3.h"
#include "math/mat4.h"
#include "math/ray.h"
#include "math/simd.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace xn;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal test framework
// ─────────────────────────────────────────────────────────────────────────────
static int s_pass = 0, s_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL  %s:%d — %s\n", __FILE__, __LINE__, #expr); \
        ++s_fail; \
    } else { ++s_pass; } \
} while(0)

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a)-(b)) < (eps))

static void run(const char* name, std::function<void()> fn) {
    std::printf("[ ] %s\n", name);
    fn();
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec3 tests
// ─────────────────────────────────────────────────────────────────────────────
static void test_vec3_arithmetic() {
    Vec3 a(1,2,3), b(4,5,6);
    Vec3 sum = a + b;
    CHECK_NEAR(sum.x, 5.f, 1e-6f);
    CHECK_NEAR(sum.y, 7.f, 1e-6f);
    CHECK_NEAR(sum.z, 9.f, 1e-6f);

    Vec3 diff = b - a;
    CHECK_NEAR(diff.x, 3.f, 1e-6f);

    Vec3 scaled = a * 2.f;
    CHECK_NEAR(scaled.z, 6.f, 1e-6f);
}

static void test_vec3_dot_cross() {
    Vec3 x(1,0,0), y(0,1,0), z(0,0,1);
    CHECK_NEAR(dot(x, y), 0.f, 1e-6f);
    CHECK_NEAR(dot(x, x), 1.f, 1e-6f);

    Vec3 c = cross(x, y);
    CHECK_NEAR(c.x, 0.f, 1e-6f);
    CHECK_NEAR(c.y, 0.f, 1e-6f);
    CHECK_NEAR(c.z, 1.f, 1e-6f);
}

static void test_vec3_normalize() {
    Vec3 v(3, 4, 0);
    Vec3 n = normalize(v);
    CHECK_NEAR(n.length(), 1.f, 1e-6f);
    CHECK_NEAR(n.x, 0.6f, 1e-5f);
    CHECK_NEAR(n.y, 0.8f, 1e-5f);
}

static void test_vec3_reflect_refract() {
    Vec3 n(0, 1, 0);
    Vec3 d = normalize(Vec3(1, -1, 0));
    Vec3 r = reflect(d, n);
    CHECK_NEAR(r.x, d.x, 1e-5f);
    CHECK_NEAR(r.y, -d.y, 1e-5f);

    Vec3 out;
    bool ok = refract(-n, n, 1.f / 1.5f, out);
    CHECK(ok);
    CHECK_NEAR(out.length(), 1.f, 1e-4f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mat4 tests
// ─────────────────────────────────────────────────────────────────────────────
static void test_mat4_identity_mul() {
    Mat4 I = Mat4::identity();
    Vec3 p(1, 2, 3);
    Vec3 tp = I.transform_point(p);
    CHECK_NEAR(tp.x, 1.f, 1e-5f);
    CHECK_NEAR(tp.y, 2.f, 1e-5f);
    CHECK_NEAR(tp.z, 3.f, 1e-5f);
}

static void test_mat4_translate() {
    Mat4 T = Mat4::translate(Vec3(5, 3, 1));
    Vec3 p = T.transform_point(Vec3(0, 0, 0));
    CHECK_NEAR(p.x, 5.f, 1e-5f);
    CHECK_NEAR(p.y, 3.f, 1e-5f);
    CHECK_NEAR(p.z, 1.f, 1e-5f);

    // translate should not affect directions
    Vec3 d = T.transform_dir(Vec3(1, 0, 0));
    CHECK_NEAR(d.x, 1.f, 1e-5f);
    CHECK_NEAR(d.y, 0.f, 1e-5f);
}

static void test_mat4_inverse() {
    Mat4 T = Mat4::translate(Vec3(1, 2, 3));
    Mat4 Ti = T.inverse();
    Mat4 I = T * Ti;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float expected = (i == j) ? 1.f : 0.f;
            CHECK_NEAR(I.m[i][j], expected, 1e-4f);
        }
}

static void test_mat4_rotate() {
    // 90deg rotation around Z should map X→Y
    Mat4 R = Mat4::rotate(Vec3(0,0,1), kPi * 0.5f);
    Vec3 px = R.transform_dir(Vec3(1,0,0));
    CHECK_NEAR(px.x, 0.f, 1e-5f);
    CHECK_NEAR(px.y, 1.f, 1e-5f);
    CHECK_NEAR(px.z, 0.f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD Float4 tests
// ─────────────────────────────────────────────────────────────────────────────
static void test_float4_arithmetic() {
    Float4 a(1.f, 2.f, 3.f, 4.f);
    Float4 b(5.f, 6.f, 7.f, 8.f);
    Float4 c = a + b;
    CHECK_NEAR(c[0], 6.f,  1e-5f);
    CHECK_NEAR(c[1], 8.f,  1e-5f);
    CHECK_NEAR(c[2], 10.f, 1e-5f);
    CHECK_NEAR(c[3], 12.f, 1e-5f);

    Float4 d = a * b;
    CHECK_NEAR(d[0], 5.f, 1e-5f);
    CHECK_NEAR(d[3], 32.f, 1e-5f);
}

static void test_float4_fmadd() {
    Float4 a(2.f), b(3.f), c(1.f);
    Float4 r = fmadd4(a, b, c);
    CHECK_NEAR(r[0], 7.f, 1e-5f);
    CHECK_NEAR(r[3], 7.f, 1e-5f);
}

static void test_float4_hadd() {
    Float4 a(1.f, 2.f, 3.f, 4.f);
    float s = hadd4(a);
    CHECK_NEAR(s, 10.f, 1e-5f);
}

static void test_float4_minmax() {
    Float4 a(1.f, 5.f, 3.f, 2.f);
    Float4 b(4.f, 2.f, 4.f, 1.f);
    Float4 mn = min4(a, b);
    Float4 mx = max4(a, b);
    CHECK_NEAR(mn[0], 1.f, 1e-5f);
    CHECK_NEAR(mn[1], 2.f, 1e-5f);
    CHECK_NEAR(mx[0], 4.f, 1e-5f);
    CHECK_NEAR(mx[1], 5.f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Float8 tests
// ─────────────────────────────────────────────────────────────────────────────
static void test_float8_arithmetic() {
    Float8 a(2.f), b(3.f);
    Float8 c = a * b;
    for (int i = 0; i < 8; ++i)
        CHECK_NEAR(((float*)&c.v)[i], 6.f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Onb tests
// ─────────────────────────────────────────────────────────────────────────────
static void test_onb() {
    Vec3 normals[] = { Vec3(0,1,0), Vec3(1,0,0), Vec3(0,0,1), normalize(Vec3(1,1,1)) };
    for (auto n : normals) {
        Onb onb(n);
        // Check orthonormality
        CHECK_NEAR(dot(onb.u, onb.v), 0.f, 1e-5f);
        CHECK_NEAR(dot(onb.u, onb.n), 0.f, 1e-5f);
        CHECK_NEAR(dot(onb.v, onb.n), 0.f, 1e-5f);
        CHECK_NEAR(onb.u.length(), 1.f, 1e-5f);
        CHECK_NEAR(onb.v.length(), 1.f, 1e-5f);
        // Round-trip: to_world(to_local(v)) == v
        Vec3 world(0.3f, 0.5f, 0.2f);
        Vec3 local = onb.to_local(world);
        Vec3 back  = onb.to_world(local);
        CHECK_NEAR(back.x, world.x, 1e-4f);
        CHECK_NEAR(back.y, world.y, 1e-4f);
        CHECK_NEAR(back.z, world.z, 1e-4f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== math_test ===\n");

    run("Vec3 arithmetic",      test_vec3_arithmetic);
    run("Vec3 dot/cross",       test_vec3_dot_cross);
    run("Vec3 normalize",       test_vec3_normalize);
    run("Vec3 reflect/refract", test_vec3_reflect_refract);
    run("Mat4 identity mul",    test_mat4_identity_mul);
    run("Mat4 translate",       test_mat4_translate);
    run("Mat4 rotate",          test_mat4_rotate);
    run("Mat4 inverse",         test_mat4_inverse);
    run("Float4 arithmetic",    test_float4_arithmetic);
    run("Float4 fmadd",         test_float4_fmadd);
    run("Float4 hadd",          test_float4_hadd);
    run("Float4 min/max",       test_float4_minmax);
    run("Float8 arithmetic",    test_float8_arithmetic);
    run("Onb orthonormality",   test_onb);

    std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
