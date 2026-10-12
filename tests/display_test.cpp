#include "display/color.h"
#include <cstdio>
#include <limits>
#ifdef XENON_TEST_WINDOW
#include "display/window.h"
#include <GLFW/glfw3.h>
#endif

using namespace xn;
static int failures = 0;

static void check(bool ok, const char *message) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);

	if (!ok)
		++failures;
}

int main() {
	DisplayTransform transform;
	// Independently calculated Reinhard -> sRGB reference values.
	check(transform.channel(0.f) == 0 && transform.channel(0.18f) == 109 &&
			  transform.channel(1.f) == 188 && transform.channel(2.f) == 213 &&
			  transform.channel(4.f) == 231 && transform.channel(16.f) == 248,
		  "HDR highlights remain distinguishable above one");
	check(DisplayTransform(-2.f).channel(4.f) == transform.channel(1.f) &&
			  DisplayTransform(2.f).channel(1.f) == transform.channel(4.f),
		  "exposure applies before tone mapping in photographic stops");
	check(transform.channel(-1.f) == 0 &&
			  transform.channel(std::numeric_limits<float>::max()) == 255,
		  "display conversion handles negative and extreme finite radiance");
	bool rejected = false;

	try {
		transform.channel(std::numeric_limits<float>::quiet_NaN());
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	check(rejected, "invalid radiance is not concealed by tone mapping");

#ifdef XENON_TEST_WINDOW
	// Exercises the actual OpenGL preview, including RGB row alignment,
	// vertical orientation, progressive updates, exposure and double encoding.
	for (float exposure : {0.f, -2.f}) {
		constexpr int w = 7, h = 5;
		Window window(w, h, "Xenon display regression", exposure);

		if (!window.valid())
			return 1;
		GLFWwindow *context = glfwGetCurrentContext();
		glfwHideWindow(context);
		TripleSwapchain swap(w, h);
		DisplayTransform expected(exposure);

		for (int frame = 0; frame < 2; ++frame) {
			float values[] = {0.f, 0.18f, 1.f, 2.f, 4.f, 16.f, 255.f};
			float *pixels = swap.get_write_buffer();

			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x)
					for (int c = 0; c < 3; ++c)
						pixels[(y * w + x) * 3 + c] = values[(x + y + c + frame) % 7];
			std::vector<uint8_t> encoded(w * h * 3);
			expected.encode(pixels, encoded.data(), encoded.size());
			swap.swap_writer();
			window.display(swap);
			int fw, fh;
			glfwGetFramebufferSize(context, &fw, &fh);
			std::vector<uint8_t> actual(fw * fh * 3);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadBuffer(GL_BACK);
			glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE, actual.data());
			bool match = glGetError() == GL_NO_ERROR;

			for (int y = 0; y < fh; ++y)
				for (int x = 0; x < fw; ++x)
					for (int c = 0; c < 3; ++c) {
						int sx = (x + 0.5) * w / fw, sy = h - 1 - int((y + 0.5) * h / fh);
						match &= std::abs(int(actual[(y * fw + x) * 3 + c]) -
										  int(encoded[(sy * w + sx) * 3 + c])) <= 1;
					}
			check(match, "live framebuffer matches PNG encoding and top-down orientation");
		}
	}
#endif
	return failures ? 1 : 0;
}
