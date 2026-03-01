#pragma once
// math/mat4.h — row-major 4×4 float matrix

#include "vec3.h"
#include <cstring>
#include <cmath>

namespace xn {

struct Vec4 {
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4(Vec3 v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
    Vec3 xyz() const { return {x, y, z}; }
};

struct Mat4 {
    float m[4][4]; // m[row][col]

    Mat4() { std::memset(m, 0, sizeof(m)); }

    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f;
        return r;
    }

    static Mat4 translate(Vec3 t) {
        Mat4 r = identity();
        r.m[0][3] = t.x; r.m[1][3] = t.y; r.m[2][3] = t.z;
        return r;
    }

    static Mat4 scale(Vec3 s) {
        Mat4 r = identity();
        r.m[0][0] = s.x; r.m[1][1] = s.y; r.m[2][2] = s.z;
        return r;
    }

    // Right-hand rotation around arbitrary axis (angle in radians)
    static Mat4 rotate(Vec3 axis, float angle) {
        Vec3 a = normalize(axis);
        float c = std::cos(angle), s = std::sin(angle), ic = 1.f - c;
        Mat4 r = identity();
        r.m[0][0] = a.x*a.x*ic + c;
        r.m[0][1] = a.x*a.y*ic - a.z*s;
        r.m[0][2] = a.x*a.z*ic + a.y*s;
        r.m[1][0] = a.y*a.x*ic + a.z*s;
        r.m[1][1] = a.y*a.y*ic + c;
        r.m[1][2] = a.y*a.z*ic - a.x*s;
        r.m[2][0] = a.z*a.x*ic - a.y*s;
        r.m[2][1] = a.z*a.y*ic + a.x*s;
        r.m[2][2] = a.z*a.z*ic + c;
        return r;
    }

    // Perspective projection (right-hand, depth [-1,1] OpenGL style)
    static Mat4 perspective(float fov_y_rad, float aspect, float near, float far) {
        float f = 1.f / std::tan(fov_y_rad * 0.5f);
        float d = far - near;
        Mat4 r;
        r.m[0][0] = f / aspect;
        r.m[1][1] = f;
        r.m[2][2] = -(far + near) / d;
        r.m[2][3] = -(2.f * far * near) / d;
        r.m[3][2] = -1.f;
        return r;
    }

    // Look-at (right-hand)
    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 f = normalize(target - eye);
        Vec3 r = normalize(cross(f, up));
        Vec3 u = cross(r, f);
        Mat4 m = identity();
        m.m[0][0]=r.x; m.m[0][1]=r.y; m.m[0][2]=r.z; m.m[0][3]=-dot(r,eye);
        m.m[1][0]=u.x; m.m[1][1]=u.y; m.m[1][2]=u.z; m.m[1][3]=-dot(u,eye);
        m.m[2][0]=-f.x;m.m[2][1]=-f.y;m.m[2][2]=-f.z;m.m[2][3]= dot(f,eye);
        m.m[3][3]=1.f;
        return m;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }

    Vec4 operator*(Vec4 v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w
        };
    }

    Vec3 transform_point(Vec3 p) const {
        Vec4 r = (*this) * Vec4(p, 1.f);
        return r.xyz() * (1.f / r.w);
    }

    Vec3 transform_dir(Vec3 d) const {
        return ((*this) * Vec4(d, 0.f)).xyz();
    }

    Mat4 transposed() const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }

    // Inverse via cofactor expansion (general 4×4)
    Mat4 inverse() const;
};

// Implemented in mat4 itself to keep header-only
inline Mat4 Mat4::inverse() const {
    // Gauss-Jordan elimination
    float a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = m[i][j];
        for (int j = 0; j < 4; ++j) a[i][j+4] = (i==j) ? 1.f : 0.f;
    }
    for (int col = 0; col < 4; ++col) {
        // Partial pivot
        int pivot = col;
        for (int row = col+1; row < 4; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
        for (int k = 0; k < 8; ++k) std::swap(a[col][k], a[pivot][k]);
        float inv_diag = 1.f / a[col][col];
        for (int k = 0; k < 8; ++k) a[col][k] *= inv_diag;
        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            float factor = a[row][col];
            for (int k = 0; k < 8; ++k) a[row][k] -= factor * a[col][k];
        }
    }
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r.m[i][j] = a[i][j+4];
    return r;
}

} // namespace xn
