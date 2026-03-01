#include "material/bsdf.h"
#include "math/ray.h"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace xn;

void test_bsdf_consistency() {
    std::printf("--- BSDF Consistency Test (Monte Carlo) ---\n");
    
    PCGState rng = seed_pcg(42);
    Vec3 wo = normalize(Vec3(0.5, 0.5, 1.0)); // Off-axis incidence (checks Fresnel)
    int samples = 1000000;

    // 1. Diffuse Test
    {
        PrincipledBSDF mat;
        mat.albedo = Vec3(0.5f, 0.8f, 0.3f);
        mat.roughness = 1.0f;
        mat.metallic = 0.0f;

        Vec3 sum(0.f);
        for (int i = 0; i < samples; ++i) {
            BSDFSample s;
            if (bsdf_sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)samples;
        std::printf("Diffuse reflectance: [%.3f, %.3f, %.3f] (expected ~[0.500, 0.800, 0.300])\n", 
            avg.x, avg.y, avg.z);
    }

    // 2. Glossy Metallic Test
    {
        PrincipledBSDF mat;
        mat.albedo = Vec3(1.0f, 1.0f, 1.0f);
        mat.roughness = 0.1f;
        mat.metallic = 1.0f;

        Vec3 sum(0.f);
        for (int i = 0; i < samples; ++i) {
            BSDFSample s;
            if (bsdf_sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)samples;
        std::printf("Glossy Metallic reflectance: [%.3f, %.3f, %.3f] (expected ~[1.000, 1.000, 1.000])\n", 
            avg.x, avg.y, avg.z);
    }

    // 3. Rough Plastic Test
    {
        PrincipledBSDF mat;
        mat.albedo = Vec3(0.5f, 0.5f, 0.5f);
        mat.roughness = 0.5f;
        mat.metallic = 0.0f;
        mat.specular = 0.5f; // f0 ~ 0.04

        Vec3 sum(0.f);
        for (int i = 0; i < samples; ++i) {
            BSDFSample s;
            if (bsdf_sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)samples;
        // Plastic is roughly diffuse albedo + specular reflectance (~0.04 at normal incidence)
        std::printf("Rough Plastic reflectance: [%.3f, %.3f, %.3f] (expected ~[0.540, 0.540, 0.540])\n", 
            avg.x, avg.y, avg.z);
    }
}

int main() {
    test_bsdf_consistency();
    return 0;
}
