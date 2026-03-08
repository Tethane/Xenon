#include "display/window.h"
#include "render/swapchain.h"
#include "render/wavefront.h"
#include "scene/scene_file.h"
#include "math/image.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>
#include <iostream>

#ifdef XENON_HAS_CUDA
#include "cuda/cuda_renderer.cuh"
#endif

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

  // Noise Computation

  NoiseStats s = ImageNoiseEstimator::computeResidualNoiseStats(data, w, h);

  std::cout << std::setprecision(10)
                << "meanSquaredScore = " << s.meanSquaredScore << "\n"
                << "rmsScore         = " << s.rmsScore << "\n"
                << "meanAbsolute     = " << s.meanAbsolute << "\n"
                << "medianAbsolute   = " << s.medianAbsolute << "\n"
                << "p95Absolute      = " << s.p95Absolute << "\n"
                << "maxAbsolute      = " << s.maxAbsolute << "\n"
                << "avgLuminance     = " << s.avgLuminance << "\n"
                << "edgeThreshold    = " << s.edgeThreshold << "\n"
                << "epsilon          = " << s.epsilon << "\n"
                << "validPixelCount  = " << s.validPixelCount << "\n"
                << "validPixelRatio  = " << s.validPixelRatio << "\n";

  std::printf("Saved image to %s\n", path.c_str());
}

int main(int argc, char* argv[]) {
  // Setup Scene and BVH Construction
  SceneConfig config;
  
  // Default scene path (can be overridden by args)
  std::string scene_path = "";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--scene" && i + 1 < argc) {
      scene_path = argv[++i];
    } else if (argv[i][0] != '-' && scene_path.empty()) {
      scene_path = argv[i];
    }
  }
  if (scene_path.empty()) scene_path = "scenes/cornell_box.xenon";

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
    } else if (std::string(argv[i]) == "--backend" && i + 1 < argc) {
      config.backend = argv[++i];
    }
  }

  std::string final_output = output_dir + "/" + config.output;

  // Begin Timing (for aggregate performance)
  auto start = std::chrono::high_resolution_clock::now();

  // Determine backend
  bool use_cuda = false;
#ifdef XENON_HAS_CUDA
  if (config.backend == "cuda") {
    use_cuda = true;
    std::printf("[Backend] CUDA GPU wavefront renderer\n");
  } else {
    std::printf("[Backend] CPU tiled renderer\n");
  }
#else
  if (config.backend == "cuda") {
    std::printf("[Warning] CUDA backend requested but not available — falling back to CPU\n");
  }
  std::printf("[Backend] CPU tiled renderer\n");
#endif

  std::printf("[Config] %dx%d, %d spp, bounces %d-%d\n",
              config.width, config.height, config.samples, config.min_bounces, config.max_bounces);

  // Setup window, OpenGL shaders for display, swapchain, threadpool for wavefront
  Window win(config.width, config.height, "Xenon (" + scene_path + ")");
  TripleSwapchain swapchain(config.width, config.height);

  // Create appropriate renderer
  std::unique_ptr<WavefrontRenderer> cpu_renderer;
  int current_spp = 0;

#ifdef XENON_HAS_CUDA
  std::unique_ptr<CudaRenderer> cuda_renderer;
  if (use_cuda) {
    cuda_renderer = std::make_unique<CudaRenderer>(
      config.width, config.height, config.max_bounces, config.min_bounces);
    cuda_renderer->upload_scene(scene);
  } else {
    cpu_renderer = std::make_unique<WavefrontRenderer>(config.width, config.height);
  }
#else
  cpu_renderer = std::make_unique<WavefrontRenderer>(config.width, config.height);
#endif

  // Render Asynchronously
  std::atomic<bool> exit_flag(false);
  std::thread render_thread([&]() {
    while (!exit_flag.load()) {
      if (current_spp < config.samples) {
#ifdef XENON_HAS_CUDA
        if (use_cuda) {
          cuda_renderer->render_frame(scene, camera, swapchain);
          current_spp = cuda_renderer->spp();
        } else {
          cpu_renderer->render_frame_tiled(scene, camera, swapchain);
          current_spp = cpu_renderer->get_spp();
        }
#else
        cpu_renderer->render_frame_tiled(scene, camera, swapchain);
        current_spp = cpu_renderer->get_spp();
#endif
        if (current_spp % 10 == 0) {
          std::printf("Progress: %d/%d samples\n", current_spp, config.samples);
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  });

  // Display on Main Thread
  while (!win.should_close()) {
    win.poll_events();
    win.display(swapchain);
    win.swap_buffers();
    
    if (current_spp >= config.samples) {
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
