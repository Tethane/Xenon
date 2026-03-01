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

void save_image(const std::string& path, int w, int h, float* data) {
    std::vector<uint8_t> pixels(w * h * 3);
    for (int i = 0; i < w * h * 3; ++i) {
        float c = std::pow(std::clamp(data[i], 0.f, 1.f), 1.f/2.2f); // Gamma
        pixels[i] = (uint8_t)(255.99f * c);
    }
    stbi_write_png(path.c_str(), w, h, 3, pixels.data(), w * 3);
    std::printf("Saved image to %s\n", path.c_str());
}

int main(int argc, char** argv) {
    SceneConfig config;
    std::string scene_path = "scenes/cornell_box.xenon";
    
    // Initial scene path (can be overridden by args)
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
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--samples" && i + 1 < argc) {
            config.samples = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            config.output = argv[++i];
        } else if (std::string(argv[i]) == "--min-bounces" && i + 1 < argc) {
            config.min_bounces = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--max-bounces" && i + 1 < argc) {
            config.max_bounces = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    std::string final_output = output_dir + "/" + config.output;

    Window win(config.width, config.height, "Xenon Path Tracer");
    TripleSwapchain swap(config.width, config.height);
    WavefrontRenderer renderer(config.width, config.height);
    renderer.set_bounces(config.min_bounces, config.max_bounces);

    std::atomic<bool> exit_flag(false);
    std::thread render_thread([&]() {
        while (!exit_flag.load()) {
            if (renderer.get_spp() < config.samples) {
                renderer.render_frame(scene, camera, swap);
                if (renderer.get_spp() % 10 == 0) {
                    std::printf("Progress: %d/%d samples\n", renderer.get_spp(), config.samples);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });

    while (!win.should_close()) {
        win.poll_events();
        win.display(swap);
        win.swap_buffers();
        
        if (renderer.get_spp() >= config.samples) {
            // Finished!
            save_image(final_output, config.width, config.height, swap.get_read_buffer());
            break;
        }
    }

    exit_flag.store(true);
    render_thread.join();
    return 0;
}
