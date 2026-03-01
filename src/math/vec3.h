#pragma once
// math/vec3.h — 3-component float vector used throughout the renderer
// All operations are scalar but designed to be hoisted into SIMD contexts.

#include <cmath>
#include <algorithm>
#include <cassert>
#include <immintrin.h>

namespace xn {

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    explicit Vec3(float s) : x(s), y(s), z(s) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float  operator[](int i) const { return (&x)[i]; }
    float& operator[](int i)       { return (&x)[i]; }

    Vec3 operator+(Vec3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(Vec3 b) const { return {x*b.x, y*b.y, z*b.z}; }
    Vec3 operator/(Vec3 b) const { return {x/b.x, y/b.y, z/b.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { float r=1.f/s; return {x*r,y*r,z*r}; }
    Vec3 operator-() const { return {-x,-y,-z}; }

    Vec3& operator+=(Vec3 b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(Vec3 b) { x-=b.x; y-=b.y; z-=b.z; return *this; }
    Vec3& operator*=(Vec3 b) { x*=b.x; y*=b.y; z*=b.z; return *this; }
    Vec3& operator*=(float s){ x*=s;   y*=s;   z*=s;   return *this; }
    Vec3& operator/=(float s){ *this *= (1.f/s); return *this; }

    bool operator==(Vec3 b) const { return x==b.x && y==b.y && z==b.z; }
    bool operator!=(Vec3 b) const { return !(*this==b); }

    float length_sq() const { return x*x + y*y + z*z; }
    float length()    const { return std::sqrt(length_sq()); }
    bool  near_zero() const { constexpr float e=1e-8f; return std::abs(x)<e && std::abs(y)<e && std::abs(z)<e; }
};

inline Vec3 operator*(float s, Vec3 v) { return v*s; }

inline float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3  cross(Vec3 a, Vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
inline Vec3  normalize(Vec3 v) { return v / v.length(); }
inline Vec3  reflect(Vec3 v, Vec3 n) { return v - 2.f*dot(v,n)*n; }
inline Vec3  faceforward(Vec3 n, Vec3 ref) { return dot(n,ref)<0.f ? -n : n; }
inline Vec3  lerp(Vec3 a, Vec3 b, float t) { return a + t*(b-a); }
inline Vec3  min3(Vec3 a, Vec3 b) { return {std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z)}; }
inline Vec3  max3(Vec3 a, Vec3 b) { return {std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z)}; }
inline Vec3  abs3(Vec3 v) { return {std::abs(v.x), std::abs(v.y), std::abs(v.z)}; }
inline Vec3  clamp3(Vec3 v, float lo, float hi) {
    return {std::clamp(v.x,lo,hi), std::clamp(v.y,lo,hi), std::clamp(v.z,lo,hi)};
}
// Component-wise refraction (Snell's law). Returns false if TIR.
inline bool refract(Vec3 v, Vec3 n, float ni_over_nt, Vec3& out) {
    float cos_i = -dot(normalize(v), n);
    float sin2_t = ni_over_nt*ni_over_nt*(1.f - cos_i*cos_i);
    if (sin2_t >= 1.f) return false;
    out = ni_over_nt*v + (ni_over_nt*cos_i - std::sqrt(1.f - sin2_t))*n;
    return true;
}
inline float max_component(Vec3 v) { return std::max({v.x, v.y, v.z}); }
inline int   max_axis(Vec3 v) {
    return (v.x>v.y && v.x>v.z) ? 0 : (v.y>v.z ? 1 : 2);
}

// Convenience color aliases  (Vec3 doubles as RGB)
inline Vec3 linear_to_srgb(Vec3 c) {
    auto cvt = [](float x) -> float {
        x = std::clamp(x, 0.f, 1.f);
        return x <= 0.0031308f ? 12.92f*x : 1.055f*std::pow(x, 1.f/2.2f) - 0.055f;
    };
    return {cvt(c.x), cvt(c.y), cvt(c.z)};
}

} // namespace xn
