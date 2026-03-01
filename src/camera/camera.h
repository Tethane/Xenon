#pragma once
// camera/camera.h — Perspective camera with FOV and transform

#include "math/vec3.h"
#include "math/ray.h"
#include "math/mat4.h"

namespace xn {

class Camera {
public:
    Camera() = default;

    // Initialize camera with position, forward, up, and vertical FOV (degrees)
    void look_at(Vec3 eye, Vec3 target, Vec3 up, float fov_y_deg, float aspect) {
        eye_ = eye;
        
        float theta = fov_y_deg * kPi / 180.f;
        float h = std::tan(theta / 2.f);
        float viewport_height = 2.f * h;
        float viewport_width = aspect * viewport_height;

        // Camera coordinate system (right-hand)
        // w is backward, u is right, v is up
        w_ = normalize(eye - target);
        u_ = normalize(cross(up, w_));
        v_ = cross(w_, u_);

        horizontal_ = viewport_width * u_;
        vertical_ = viewport_height * v_;
        lower_left_corner_ = eye_ - horizontal_ * 0.5f - vertical_ * 0.5f - w_;
    }

    // Generate a ray for a given (u, v) in [0, 1] screen space
    Ray get_ray(float s, float t) const {
        return Ray{eye_, normalize(lower_left_corner_ + s * horizontal_ + t * vertical_ - eye_)};
    }

private:
    Vec3 eye_;
    Vec3 lower_left_corner_;
    Vec3 horizontal_;
    Vec3 vertical_;
    Vec3 u_, v_, w_;
};

} // namespace xn
