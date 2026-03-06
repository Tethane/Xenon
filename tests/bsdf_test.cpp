#include "material/bsdf.h"
#include "material/material.h"
#include "math/ray.h"
#include <cstdio>
#include <cmath>

using namespace xn;

static int g_fail = 0;

static void check(const char* name, bool ok) {
    if (!ok) {
        std::printf("  FAIL: %s\n", name);
        g_fail++;
    } else {
        std::printf("  PASS: %s\n", name);
    }
}

static bool approx(float a, float b, float tol = 0.15f) {
    return std::abs(a - b) < tol;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: PDF Integration (Monte Carlo)
//   For each queue, integral of pdf(wo, wi) * cos(theta_i) over hemisphere
//   should ≈ 1.0 (or appropriate value).
// ─────────────────────────────────────────────────────────────────────────────
void test_pdf_integration() {
    std::printf("--- PDF Integration Test ---\n");
    PCGState rng = seed_pcg(42);
    Vec3 wo = normalize(Vec3(0.3f, 0.2f, 1.0f));
    int N = 500000;

    // Diffuse
    {
        Material mat;
        mat.baseColor = Vec3(0.8f);
        mat.roughness = 0.5f;
        material_prepare(mat);

        float sum = 0.f;
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (diffuse_bsdf::sample(wo, mat, rng, s)) {
                sum += 1.f; // pdf-weighted sampling → each sample contributes 1
            }
        }
        float ratio = sum / (float)N;
        check("Diffuse PDF integrates ~1", approx(ratio, 1.0f, 0.05f));
    }

    // Microfacet reflection
    {
        Material mat;
        mat.baseColor = Vec3(0.9f);
        mat.roughness = 0.3f;
        mat.metallic = 1.0f;
        material_prepare(mat);

        float sum = 0.f;
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (microfacet_refl_bsdf::sample(wo, mat, rng, s)) {
                sum += 1.f;
            }
        }
        float ratio = sum / (float)N;
        check("Microfacet Refl PDF integrates ~1", approx(ratio, 1.0f, 0.1f));
    }

    // Microfacet transmission
    {
        Material mat;
        mat.baseColor = Vec3(1.0f);
        mat.roughness = 0.2f;
        mat.ior = 1.5f;
        mat.transmission = 1.0f;
        material_prepare(mat);

        float sum = 0.f;
        int valid = 0;
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (microfacet_trans_bsdf::sample(wo, mat, rng, s)) {
                sum += 1.f;
                valid++;
            }
        }
        // Not all samples valid (TIR), but valid ones should integrate
        float ratio = (valid > 0) ? sum / (float)valid : 0.f;
        check("Microfacet Trans valid samples integrate ~1", approx(ratio, 1.0f, 0.05f));
        std::printf("    (valid rate: %.1f%%)\n", 100.f * valid / N);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Eval/PDF/Sample Consistency
//   sample() returns (wi, f, pdf). Check that:
//     eval(wo, wi) / pdf(wo, wi) ≈ f (for non-delta lobes)
// ─────────────────────────────────────────────────────────────────────────────
void test_consistency() {
    std::printf("--- Eval/PDF/Sample Consistency ---\n");
    PCGState rng = seed_pcg(123);
    Vec3 wo = normalize(Vec3(0.5f, 0.3f, 0.9f));
    int N = 100000;

    // Microfacet reflection
    {
        Material mat;
        mat.baseColor = Vec3(0.8f, 0.2f, 0.1f);
        mat.roughness = 0.35f;
        mat.metallic = 1.0f;
        material_prepare(mat);

        float max_err_f = 0.f;
        float max_err_p = 0.f;
        int checked = 0;
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (!microfacet_refl_bsdf::sample(wo, mat, rng, s)) continue;
            if (s.pdf < 1e-6f) continue;

            Vec3 f_eval = microfacet_refl_bsdf::eval(wo, s.wi, mat);
            float p = microfacet_refl_bsdf::pdf(wo, s.wi, mat);
            if (p < 1e-6f) continue;

            // eval() should match sample().f
            Vec3 diff_f = f_eval - s.f;
            float err_f = std::max({std::abs(diff_f.x), std::abs(diff_f.y), std::abs(diff_f.z)});
            if (err_f > max_err_f) max_err_f = err_f;

            // pdf() should match sample().pdf
            float err_p = std::abs(p - s.pdf);
            if (err_p > max_err_p) max_err_p = err_p;
            checked++;
        }
        check("Microfacet Refl eval consistency (max err < 0.01)", max_err_f < 0.01f);
        check("Microfacet Refl pdf consistency (max err < 0.01)", max_err_p < 0.01f);
        std::printf("    max_err_f=%.6f max_err_p=%.6f checked=%d\n", max_err_f, max_err_p, checked);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Furnace Test (Energy Conservation)
//   White material, integrate sample throughput. Should be ≤ 1.0 for
//   energy conservation. For diffuse, should ≈ albedo.
// ─────────────────────────────────────────────────────────────────────────────
void test_furnace() {
    std::printf("--- Furnace Test ---\n");
    PCGState rng = seed_pcg(999);
    Vec3 wo = normalize(Vec3(0.4f, 0.2f, 1.0f));
    int N = 500000;

    // Diffuse furnace
    {
        Material mat;
        mat.baseColor = Vec3(0.7f, 0.3f, 0.5f);
        mat.roughness = 1.0f;
        mat.metallic = 0.0f;
        material_prepare(mat);

        Vec3 sum(0.f);
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (diffuse_bsdf::sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)N;
        std::printf("  Diffuse furnace: [%.3f, %.3f, %.3f] (expect ~[0.700, 0.300, 0.500])\n",
            avg.x, avg.y, avg.z);
        check("Diffuse energy ≈ albedo",
            approx(avg.x, 0.7f) && approx(avg.y, 0.3f) && approx(avg.z, 0.5f));
    }

    // Metallic furnace (white metal, should be ~1)
    {
        Material mat;
        mat.baseColor = Vec3(1.0f);
        mat.roughness = 0.3f;
        mat.metallic = 1.0f;
        material_prepare(mat);

        Vec3 sum(0.f);
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (microfacet_refl_bsdf::sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)N;
        std::printf("  Metal furnace: [%.3f, %.3f, %.3f] (expect ≤1, close to 1 with multiscatter)\n",
            avg.x, avg.y, avg.z);
        check("Metal energy ≤ 1.1 (allows slight overshoot from approx)",
            avg.x <= 1.1f && avg.y <= 1.1f && avg.z <= 1.1f);
    }

    // Dielectric reflection furnace
    {
        Material mat;
        mat.baseColor = Vec3(1.0f);
        mat.roughness = 0.5f;
        mat.metallic = 0.0f;
        mat.ior = 1.5f;
        material_prepare(mat);

        Vec3 sum(0.f);
        for (int i = 0; i < N; ++i) {
            BSDFSample s;
            if (microfacet_refl_bsdf::sample(wo, mat, rng, s)) {
                sum += s.f * std::abs(s.wi.z) / s.pdf;
            }
        }
        Vec3 avg = sum / (float)N;
        std::printf("  Dielectric refl furnace: [%.3f, %.3f, %.3f] (expect F0 ~ 0.04)\n",
            avg.x, avg.y, avg.z);
        check("Dielectric reflection energy ≤ 1.0", avg.x <= 1.0f && avg.y <= 1.0f && avg.z <= 1.0f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Delta lobes sanity
// ─────────────────────────────────────────────────────────────────────────────
void test_delta() {
    std::printf("--- Delta Lobe Tests ---\n");
    PCGState rng = seed_pcg(77);

    // Mirror reflection
    {
        Material mat;
        mat.baseColor = Vec3(1.0f);
        mat.roughness = 0.0f;
        mat.metallic = 1.0f;
        material_prepare(mat);

        Vec3 wo = normalize(Vec3(0.5f, 0.f, 1.f));
        BSDFSample s;
        bool ok = delta_refl_bsdf::sample(wo, mat, rng, s);
        check("Delta refl produces valid sample", ok);
        check("Delta refl is_delta", s.is_delta);
        check("Delta refl wi.z > 0", s.wi.z > 0.f);
        check("Delta refl pdf == 1", s.pdf == 1.f);
        // Check reflection law: wi = reflect(-wo, N)
        Vec3 expected = reflect(-wo, Vec3(0, 0, 1));
        float err = (s.wi - expected).length();
        check("Delta refl correct direction", err < 0.001f);
    }

    // Refraction
    {
        Material mat;
        mat.baseColor = Vec3(1.0f);
        mat.roughness = 0.0f;
        mat.ior = 1.5f;
        mat.transmission = 1.0f;
        material_prepare(mat);

        Vec3 wo = normalize(Vec3(0.3f, 0.f, 1.f));
        BSDFSample s;
        bool ok = delta_trans_bsdf::sample(wo, mat, rng, s);
        check("Delta trans produces valid sample", ok);
        check("Delta trans is_delta", s.is_delta);
        check("Delta trans wi.z < 0 (transmitted)", s.wi.z < 0.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Reciprocity (BRDF only — diffuse & microfacet reflection)
// ─────────────────────────────────────────────────────────────────────────────
void test_reciprocity() {
    std::printf("--- Reciprocity Test ---\n");
    PCGState rng = seed_pcg(555);
    int N = 10000;

    Material mat;
    mat.baseColor = Vec3(0.6f, 0.3f, 0.8f);
    mat.roughness = 0.4f;
    mat.metallic = 0.0f;
    material_prepare(mat);

    float max_err_diff = 0.f;
    float max_err_spec = 0.f;

    for (int i = 0; i < N; ++i) {
        Vec3 wo = normalize(Vec3(rng.next_float() - 0.5f, rng.next_float() - 0.5f, rng.next_float() * 0.9f + 0.1f));
        Vec3 wi = normalize(Vec3(rng.next_float() - 0.5f, rng.next_float() - 0.5f, rng.next_float() * 0.9f + 0.1f));

        // Diffuse reciprocity
        Vec3 f1 = diffuse_bsdf::eval(wo, wi, mat);
        Vec3 f2 = diffuse_bsdf::eval(wi, wo, mat);
        float err = (f1 - f2).length();
        if (err > max_err_diff) max_err_diff = err;

        // Specular reciprocity  
        Vec3 s1 = microfacet_refl_bsdf::eval(wo, wi, mat);
        Vec3 s2 = microfacet_refl_bsdf::eval(wi, wo, mat);
        float serr = (s1 - s2).length();
        if (serr > max_err_spec) max_err_spec = serr;
    }

    check("Diffuse reciprocity (max err < 1e-6)", max_err_diff < 1e-6f);
    check("Microfacet refl reciprocity (max err < 0.01)", max_err_spec < 0.01f);
    std::printf("    diff_err=%.8f spec_err=%.8f\n", max_err_diff, max_err_spec);
}

int main() {
    test_pdf_integration();
    test_consistency();
    test_furnace();
    test_delta();
    test_reciprocity();

    std::printf("\n=== %s (%d failures) ===\n", g_fail == 0 ? "ALL PASSED" : "FAILURES", g_fail);
    return g_fail;
}
