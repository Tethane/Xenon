#pragma once
#include <memory>
#include <string>
#include <vector>
#include "math/vec3.h"

namespace xn {
// Pixels are linear-light RGBA; level zero rows follow the image's top-to-bottom
// order. UV (0,0) is the bottom-left, independent of the file format.
struct TexturePixel {
	float r = 0, g = 0, b = 0, a = 1;

	Vec3 rgb() const {
		return {r, g, b};
	}

	float channel(int c) const {
		return c == 1 ? g : c == 2 ? b : c == 3 ? a : r;
	}
};

struct TextureLevel {
	int width = 0, height = 0, offset = 0;
};

struct Texture {
	std::string path;
	bool srgb = false;
	std::vector<TextureLevel> levels;
	std::vector<TexturePixel> pixels;
	TexturePixel sample(float u, float v, float lod = 0.f) const;
	void generate_mips();
};

std::shared_ptr<const Texture> load_texture(const std::string &path, bool srgb);

struct TextureMap {
	std::shared_ptr<const Texture> image;
	int channel = 0;

	explicit operator bool() const {
		return bool(image);
	}

	TexturePixel sample(float u, float v, float lod = 0.f) const {
		return image ? image->sample(u, v, lod) : TexturePixel{1, 1, 1, 1};
	}

	float scalar(float u, float v, float lod = 0.f) const {
		return sample(u, v, lod).channel(channel);
	}
};
} // namespace xn
