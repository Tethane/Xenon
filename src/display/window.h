#pragma once
// display/window.h — GLFW window and OpenGL display logic

#include <string>
#include <vector>

#include "display/color.h"
#include "render/swapchain.h"

struct GLFWwindow;

namespace xn {

class Window {
  public:
	Window(int width, int height, const std::string &title, float exposure_ev = 0.f);
	~Window();

	bool valid() const {
		return window_ != nullptr;
	}

	bool should_close() const;
	void poll_events();

	// Uploads the swapchain's read buffer to a texture and blits to screen
	void display(TripleSwapchain &swapchain);

	void swap_buffers();

  private:
	GLFWwindow *window_ = nullptr;
	unsigned int vao_ = 0, vbo_ = 0;
	unsigned int shader_program_ = 0;
	unsigned int texture_ = 0;
	int width_, height_;
	DisplayTransform display_transform_;
	std::vector<uint8_t> display_pixels_;

	void setup_quad();
};

} // namespace xn
