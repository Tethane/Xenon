#include "geometry/mesh.h"
#include "geometry/blas.h"
#include "geometry/primitives.h"
#include "math/ray.h"
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

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a)-(b)) < (eps))

void test_ray_triangle() {
    Vec3 v0(0,0,0), v1(1,0,0), v2(0,1,0); 
    Ray r{{0.2f, 0.2f, 1.0f}, {0,0,-1}};
    float t, u, v;
    HitRecord rec;
    CHECK(ray_triangle(r, v0, v1, v2, t, u, v));
    CHECK_NEAR(t, 1.0f, 1e-5f);
    CHECK_NEAR(u, 0.2f, 1e-5f);
    CHECK_NEAR(v, 0.2f, 1e-5f);

    r.origin = {2, 2, 1}; // misses
    CHECK(!ray_triangle(r, v0, v1, v2, t, u, v));
}

void test_blas_build() {
    TriangleMesh mesh;
    // Create a simple box with 12 triangles
    mesh.add_vertex({-1,-1,-1}); // 0
    mesh.add_vertex({ 1,-1,-1}); // 1
    mesh.add_vertex({ 1, 1,-1}); // 2
    mesh.add_vertex({-1, 1,-1}); // 3
    mesh.add_vertex({-1,-1, 1}); // 4
    mesh.add_vertex({ 1,-1, 1}); // 5
    mesh.add_vertex({ 1, 1, 1}); // 6
    mesh.add_vertex({-1, 1, 1}); // 7

    // Front (z=-1)
    mesh.add_triangle(0, 1, 2); mesh.add_triangle(0, 2, 3);
    // Back (z=1)
    mesh.add_triangle(4, 5, 6); mesh.add_triangle(4, 6, 7);
    // Left
    mesh.add_triangle(0, 3, 7); mesh.add_triangle(0, 7, 4);
    // Right
    mesh.add_triangle(1, 5, 6); mesh.add_triangle(1, 6, 2);
    // Top
    mesh.add_triangle(3, 2, 6); mesh.add_triangle(3, 6, 7);
    // Bottom
    mesh.add_triangle(0, 1, 5); mesh.add_triangle(0, 5, 4);

    BLAS blas;
    blas.build(mesh);

    Ray r{{0,0,5}, {0,0,-1}};
    HitRecord rec;
    CHECK(blas.intersect(r, rec));
    CHECK_NEAR(rec.t, 4.0f, 1e-5f); // Hits back face at z=1
    CHECK_NEAR(rec.pos.z, 1.0f, 1e-5f);

    r.origin = {0,0,-5}; r.dir = {0,0,1};
    rec = {};
    CHECK(blas.intersect(r, rec));
    CHECK_NEAR(rec.t, 4.0f, 1e-5f); // Hits front face at z=-1
    CHECK_NEAR(rec.pos.z, -1.0f, 1e-5f);

    r.origin = {10,10,10}; // misses
    rec = {};
    CHECK(!blas.intersect(r, rec));
}

int main() {
    std::printf("=== geometry_test ===\n");
    test_ray_triangle();
    test_blas_build();
    std::printf("\n=== Results: %d passed, %d failed ===\n", s_pass, s_fail);
    return s_fail ? 1 : 0;
}
