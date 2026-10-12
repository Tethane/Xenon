#pragma once
// camera/camera.h — Perspective camera with FOV and transform

#include "math/vec3.h"
#include "math/ray.h"
#include "math/mat4.h"
#include <stdexcept>

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

	// Thin lens: aperture is the diameter in scene units; focus is measured
	// along the optical axis. A zero aperture preserves the pinhole sequence.
	void set_lens(float aperture, float focus_distance) {
		if (!std::isfinite(aperture) || aperture < 0.f || !std::isfinite(focus_distance) ||
			focus_distance <= 0.f)
			throw std::runtime_error(
				"Camera aperture must be nonnegative and focus distance positive");
		lens_radius_ = 0.5f * aperture;
		focus_distance_ = focus_distance;
	}

	Ray get_ray(float s, float t, float lens_u, float lens_v) const {
		if (lens_radius_ == 0.f)
			return get_ray(s, t);
		float r = lens_radius_ * std::sqrt(lens_u);
		float angle = 2.f * kPi * lens_v;
		Vec3 offset = u_ * (r * std::cos(angle)) + v_ * (r * std::sin(angle));
		Vec3 direction = lower_left_corner_ + s * horizontal_ + t * vertical_ - eye_;

		return Ray{eye_ + offset, normalize(focus_distance_ * direction - offset)};
	}

	void set_aspect(float aspect) {
		if (!std::isfinite(aspect) || aspect <= 0.f)
			throw std::runtime_error("Camera aspect must be positive");
		horizontal_ = u_ * (vertical_.length() * aspect);
		lower_left_corner_ = eye_ - horizontal_ * 0.5f - vertical_ * 0.5f - w_;
	}

	// GPU upload accessors
	Vec3 get_eye() const {
		return eye_;
	}

	Vec3 get_lower_left_corner() const {
		return lower_left_corner_;
	}

	Vec3 get_horizontal() const {
		return horizontal_;
	}

	Vec3 get_vertical() const {
		return vertical_;
	}

	Vec3 get_right() const {
		return u_;
	}

	Vec3 get_up() const {
		return v_;
	}

	float get_lens_radius() const {
		return lens_radius_;
	}

	float get_focus_distance() const {
		return focus_distance_;
	}

	void set_medium(int material_id, Vec3 absorption) {
		medium_material_ = material_id;
		absorption_ = absorption;
	}

	int get_medium_material() const {
		return medium_material_;
	}

	Vec3 get_absorption() const {
		return absorption_;
	}

  private:
	Vec3 eye_;
	Vec3 lower_left_corner_;
	Vec3 horizontal_;
	Vec3 vertical_;
	Vec3 u_, v_, w_;
	float lens_radius_ = 0.f;
	float focus_distance_ = 1.f;
	int medium_material_ = -1;
	Vec3 absorption_{};
};

} // namespace xn
