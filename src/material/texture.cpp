#include "material/texture.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace xn {
static TexturePixel blend(TexturePixel a, TexturePixel b, float t) {
	return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t,
			a.a + (b.a - a.a) * t};
}

TexturePixel Texture::sample(float u, float v, float lod) const {
	if (levels.empty())
		return {1, 1, 1, 1};
	lod = std::clamp(lod, 0.f, float(levels.size() - 1));
	int l0 = int(lod), l1 = std::min(l0 + 1, int(levels.size() - 1));
	auto bilinear = [&](int l) {
		auto &level = levels[l];
		// Reduce before integer conversion: negative and large UVs repeat safely.
		float x = (u - std::floor(u)) * level.width - 0.5f;
		float y = (1.f - (v - std::floor(v))) * level.height - 0.5f;
		int ix = int(std::floor(x)), iy = int(std::floor(y));
		float fx = x - ix, fy = y - iy;
		auto texel = [&](int px, int py) {
			px = (px % level.width + level.width) % level.width;
			py = (py % level.height + level.height) % level.height;

			return pixels[level.offset + py * level.width + px];
		};

		return blend(blend(texel(ix, iy), texel(ix + 1, iy), fx),
					 blend(texel(ix, iy + 1), texel(ix + 1, iy + 1), fx), fy);
	};

	return blend(bilinear(l0), bilinear(l1), lod - l0);
}

void Texture::generate_mips() {
	if (levels.empty())
		return;
	pixels.resize(size_t(levels[0].width) * levels[0].height);
	levels.resize(1);

	while (levels.back().width > 1 || levels.back().height > 1) {
		auto old = levels.back();
		TextureLevel next{std::max(1, (old.width + 1) / 2), std::max(1, (old.height + 1) / 2),
						  int(pixels.size())};
		// Area-weighted box resampling preserves energy for odd resolutions.
		for (int y = 0; y < next.height; ++y)
			for (int x = 0; x < next.width; ++x) {
				float x0 = float(x) * old.width / next.width,
					  x1 = float(x + 1) * old.width / next.width;
				float y0 = float(y) * old.height / next.height,
					  y1 = float(y + 1) * old.height / next.height;
				TexturePixel sum{0, 0, 0, 0};
				float total = 0;

				for (int py = int(y0); py < int(std::ceil(y1)); ++py)
					for (int px = int(x0); px < int(std::ceil(x1)); ++px) {
						float w = (std::min(x1, float(px + 1)) - std::max(x0, float(px))) *
								  (std::min(y1, float(py + 1)) - std::max(y0, float(py)));
						auto p = pixels[old.offset + std::min(py, old.height - 1) * old.width +
										std::min(px, old.width - 1)];
						sum.r += p.r * w;
						sum.g += p.g * w;
						sum.b += p.b * w;
						sum.a += p.a * w;
						total += w;
					}
				pixels.push_back({sum.r / total, sum.g / total, sum.b / total, sum.a / total});
			}
		levels.push_back(next);
	}
}

std::shared_ptr<const Texture> load_texture(const std::string &path, bool srgb) {
	static std::mutex mutex;
	static std::unordered_map<std::string, std::weak_ptr<const Texture>> cache;
	std::string canonical = std::filesystem::weakly_canonical(path).string();
	std::string key = canonical + (srgb ? "#srgb" : "#linear");
	std::lock_guard lock(mutex);

	if (auto existing = cache[key].lock())
		return existing;
	auto t = std::make_shared<Texture>();
	t->path = canonical;
	t->srgb = srgb;
	int w = 0, h = 0, n = 0;
	auto decode = [](float v) {
		return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
	};
	auto fail = [&] {
		throw std::runtime_error("Cannot load texture '" + path + "': " + stbi_failure_reason());
	};

	if (stbi_is_hdr(path.c_str())) {
		if (srgb)
			throw std::runtime_error("HDR textures must use linear color space: " + path);
		float *p = stbi_loadf(path.c_str(), &w, &h, &n, 4);

		if (!p)
			fail();
		t->pixels.resize(size_t(w) * h);

		for (size_t i = 0; i < t->pixels.size(); ++i)
			t->pixels[i] = {p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]};
		stbi_image_free(p);
	} else {
		const bool sixteen = stbi_is_16_bit(path.c_str());
		unsigned char *p8 = nullptr;
		stbi_us *p16 = nullptr;

		if (sixteen) {
			p16 = stbi_load_16(path.c_str(), &w, &h, &n, 4);

			if (!p16)
				fail();
		} else {
			p8 = stbi_load(path.c_str(), &w, &h, &n, 4);

			if (!p8)
				fail();
		}
		t->pixels.resize(size_t(w) * h);

		for (size_t i = 0; i < t->pixels.size(); ++i) {
			float v[4];

			for (int c = 0; c < 4; ++c)
				v[c] = sixteen ? p16[i * 4 + c] / 65535.f : p8[i * 4 + c] / 255.f;

			if (srgb)
				for (int c = 0; c < 3; ++c)
					v[c] = decode(v[c]);
			t->pixels[i] = {v[0], v[1], v[2], v[3]};
		}
		stbi_image_free(p8);
		stbi_image_free(p16);
	}

	for (const auto &p : t->pixels)
		for (float v : {p.r, p.g, p.b, p.a})
			if (!std::isfinite(v) || v < 0.f)
				throw std::runtime_error("Non-finite or negative texture data: " + path);
	t->levels.push_back({w, h, 0});
	t->generate_mips();
	cache[key] = t;

	return t;
}
} // namespace xn
