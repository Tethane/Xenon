#pragma once
// math/simd.h — low-level SIMD wrappers used throughout the renderer
// Requires: SSE4.2 + AVX2 (-mavx2 -mfma)

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Float4  — 4-lane SSE float
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) Float4 {
  __m128 v;

  Float4() : v(_mm_setzero_ps()) {}
  explicit Float4(float s) : v(_mm_set1_ps(s)) {}
  Float4(float a, float b, float c, float d) : v(_mm_set_ps(d, c, b, a)) {}
  explicit Float4(__m128 m) : v(m) {}

  float operator[](int i) const {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[i];
  }

  Float4 operator+(Float4 o) const { return Float4(_mm_add_ps(v, o.v)); }
  Float4 operator-(Float4 o) const { return Float4(_mm_sub_ps(v, o.v)); }
  Float4 operator*(Float4 o) const { return Float4(_mm_mul_ps(v, o.v)); }
  Float4 operator/(Float4 o) const { return Float4(_mm_div_ps(v, o.v)); }
  Float4 operator-() const { return Float4(_mm_xor_ps(v, _mm_set1_ps(-0.f))); }
  Float4 &operator+=(Float4 o) {
    v = _mm_add_ps(v, o.v);
    return *this;
  }
  Float4 &operator*=(Float4 o) {
    v = _mm_mul_ps(v, o.v);
    return *this;
  }

  // Comparison — returns mask (all bits set per lane)
  Float4 operator<(Float4 o) const { return Float4(_mm_cmplt_ps(v, o.v)); }
  Float4 operator>(Float4 o) const { return Float4(_mm_cmpgt_ps(v, o.v)); }
  Float4 operator<=(Float4 o) const { return Float4(_mm_cmple_ps(v, o.v)); }

  int movemask() const { return _mm_movemask_ps(v); }
};

inline Float4 min4(Float4 a, Float4 b) { return Float4(_mm_min_ps(a.v, b.v)); }
inline Float4 max4(Float4 a, Float4 b) { return Float4(_mm_max_ps(a.v, b.v)); }
inline Float4 fmadd4(Float4 a, Float4 b, Float4 c) {
  return Float4(_mm_fmadd_ps(a.v, b.v, c.v));
}
inline Float4 rcp4(Float4 a) {
  return Float4(_mm_div_ps(_mm_set1_ps(1.f), a.v));
}
inline Float4 sqrt4(Float4 a) { return Float4(_mm_sqrt_ps(a.v)); }
inline float hadd4(Float4 a) {
  __m128 h = _mm_hadd_ps(a.v, a.v);
  h = _mm_hadd_ps(h, h);
  return _mm_cvtss_f32(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Float8  — 8-lane AVX2 float
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(32) Float8 {
  __m256 v;

  Float8() : v(_mm256_setzero_ps()) {}
  explicit Float8(float s) : v(_mm256_set1_ps(s)) {}
  explicit Float8(__m256 m) : v(m) {}
  Float8(Float4 lo, Float4 hi)
      : v(_mm256_insertf128_ps(_mm256_castps128_ps256(lo.v), hi.v, 1)) {}

  Float8 operator+(Float8 o) const { return Float8(_mm256_add_ps(v, o.v)); }
  Float8 operator-(Float8 o) const { return Float8(_mm256_sub_ps(v, o.v)); }
  Float8 operator*(Float8 o) const { return Float8(_mm256_mul_ps(v, o.v)); }
  Float8 operator/(Float8 o) const { return Float8(_mm256_div_ps(v, o.v)); }

  Float8 operator<(Float8 o) const {
    return Float8(_mm256_cmp_ps(v, o.v, _CMP_LT_OQ));
  }
  Float8 operator>(Float8 o) const {
    return Float8(_mm256_cmp_ps(v, o.v, _CMP_GT_OQ));
  }
  Float8 operator<=(Float8 o) const {
    return Float8(_mm256_cmp_ps(v, o.v, _CMP_LE_OQ));
  }

  int movemask() const { return _mm256_movemask_ps(v); }
};

inline Float8 min8(Float8 a, Float8 b) {
  return Float8(_mm256_min_ps(a.v, b.v));
}
inline Float8 max8(Float8 a, Float8 b) {
  return Float8(_mm256_max_ps(a.v, b.v));
}
inline Float8 fmadd8(Float8 a, Float8 b, Float8 c) {
  return Float8(_mm256_fmadd_ps(a.v, b.v, c.v));
}

} // namespace xn
