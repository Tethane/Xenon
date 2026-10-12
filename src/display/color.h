#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace xn {

// Presentation only: the renderer and swapchain always contain averaged,
// scene-linear HDR radiance. Apply exposure before the nonlinear display curve.
class DisplayTransform {
  public:
	explicit DisplayTransform(float exposure_ev = 0.f) {
		if (!std::isfinite(exposure_ev) || exposure_ev < -32.f || exposure_ev > 32.f)
			throw std::runtime_error("Exposure must be finite and between -32 and 32 EV");
		scale_ = std::exp2(static_cast<double>(exposure_ev));
	}

	uint8_t channel(float radiance) const {
		if (!std::isfinite(radiance))
			throw std::runtime_error("Non-finite display radiance");
		double exposed = std::max(0.0, static_cast<double>(radiance)) * scale_;
		// Reinhard's shoulder retains HDR highlight differences instead of
		// clipping every value above one. Double avoids overflow for finite HDR.
		double mapped = exposed / (1.0 + exposed);
		double srgb =
			mapped <= 0.0031308 ? 12.92 * mapped : 1.055 * std::pow(mapped, 1.0 / 2.4) - 0.055;

		return static_cast<uint8_t>(std::clamp(std::lround(255.0 * srgb), 0L, 255L));
	}

	void encode(const float *linear, uint8_t *rgb, size_t channels) const {
		for (size_t i = 0; i < channels; ++i)
			rgb[i] = channel(linear[i]);
	}

  private:
	double scale_;
};

} // namespace xn
