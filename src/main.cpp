#include "display/window.h"
#include "render/swapchain.h"
#include "render/wavefront.h"
#include "scene/scene_file.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace xn;

// Helper for caching the image to png
void save_image(const std::string& path, int w, int h, float* data) {
  stbi_flip_vertically_on_write(true); // Fix for OpenGL coordinates

  std::vector<uint8_t> pixels(w * h * 3);
  for (int i = 0; i < w * h * 3; ++i) {
    float c = std::pow(std::clamp(data[i], 0.f, 1.f), 1.f/2.2f); // Gamma
    pixels[i] = (uint8_t)(255.99f * c);
  }
  stbi_write_png(path.c_str(), w, h, 3, pixels.data(), w * 3);
  std::printf("Saved image to %s\n", path.c_str());
}

int main(int argc, char* argv[]) {
  // Setup Scene and BVH Construction
  SceneConfig config;
  
  // Default scene path (can be overridden by args)
  std::string scene_path = "scenes/cornell_box.xenon";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
      scene_path = argv[++i];
    }
  }

  Scene scene;
  Camera camera;
  if (!load_scene(scene_path, scene, camera, config)) {
    return 1;
  }

  // Overrides from command line
  std::string output_dir = "renders";
  int render_mode = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--samples" && i + 1 < argc) {
      config.samples = std::max(8, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
      config.output = argv[++i];
    } else if (std::string(argv[i]) == "--min-bounces" && i + 1 < argc) {
      config.min_bounces = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--max-bounces" && i + 1 < argc) {
      config.max_bounces = std::max(0, std::stoi(argv[++i]));
    } else if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (std::string(argv[i]) == "--render-mode" && i + 1 < argc) {
      render_mode = std::stoi(argv[++i]);
    }
  }

  std::string final_output = output_dir + "/" + config.output;

  // Begin Timing (for aggregate performance)
  auto start = std::chrono::high_resolution_clock::now();

  // Setup window, OpenGL shaders for display, swapchain, threadpool for wavefront
  Window win(config.width, config.height, "Xenon (" + scene_path + ")");
  TripleSwapchain swapchain(config.width, config.height);
  WavefrontRenderer renderer(config.width, config.height);

  // Render Asynchronously
  std::atomic<bool> exit_flag(false);
  std::thread render_thread([&]() {
    while (!exit_flag.load()) {
      if (renderer.get_spp() < config.samples) {
        renderer.render_frame(scene, camera, swapchain);
        if (renderer.get_spp() % 10 == 0) {
          std::printf("Progress: %d/%d samples\n", renderer.get_spp(), config.samples);
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Timing for "consistent FPS" for a raytracer doesn't need to be perfect
      }
    }
  });

  // Display on Main Thread
  while (!win.should_close()) {
    win.poll_events();
    win.display(swapchain);
    win.swap_buffers();
    
    if (renderer.get_spp() >= config.samples) {
      // Finished rendering!
      save_image(final_output, config.width, config.height, swapchain.get_read_buffer());
      
      break;
    }
  }

  exit_flag.store(true);
  render_thread.join();

  // Stop Timing
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  
  std::printf("Total Render Time: %d ms\n", (int) duration.count());
  return 0;
}
