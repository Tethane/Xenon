// main.cpp — Xenon path tracer entry point

#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "display/window.h"
#include "output/measurement.h"
#include "render/swapchain.h"
#include "render/wavefront.h"
#include "scene/scene_file.h"

#ifdef XENON_HAS_CUDA
#include "cuda/cuda_renderer.cuh"
#endif

using namespace xn;

// Helper for caching the image to png
bool save_image(const std::string &path, int w, int h, const float *data, float exposure_ev) {
	stbi_flip_vertically_on_write(false); // framebuffer and PNG rows are top-down
	std::vector<uint8_t> pixels(w * h * 3);
	DisplayTransform(exposure_ev).encode(data, pixels.data(), pixels.size());
	FILE *file = std::fopen(path.c_str(), "wb");

	if (!file) {
		std::perror(path.c_str());

		return false;
	}

	struct Output {
		FILE *file;
		bool ok;
	} output{file, true};

	auto write = [](void *context, void *bytes, int size) {
		auto &out = *static_cast<Output *>(context);

		if (std::fwrite(bytes, 1, size, out.file) != static_cast<size_t>(size))
			out.ok = false;
	};
	bool encoded = stbi_write_png_to_func(write, &output, w, h, 3, pixels.data(), w * 3) != 0;
	bool closed = std::fclose(file) == 0;

	if (!encoded || !output.ok || !closed) {
		std::fprintf(stderr, "[Error] Failed to write PNG: %s\n", path.c_str());

		return false;
	}

	std::printf("Saved image to %s\n", path.c_str());

	return true;
}

static int parse_int(const std::string &value, const std::string &option) {
	int result;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

	if (error != std::errc{} || end != value.data() + value.size())
		throw std::runtime_error("Invalid integer for " + option + ": " + value);

	return result;
}

static float parse_float(const std::string &value, const std::string &option) {
	float result;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

	if (error != std::errc{} || end != value.data() + value.size())
		throw std::runtime_error("Invalid number for " + option + ": " + value);

	return result;
}

int main(int argc, char *argv[]) {
	try {
		auto total_start = MeasurementClock::now();
		RenderMeasurement metrics;
		SceneConfig config;
		std::string scene_path;
		std::string output_dir = "renders";
		bool no_display = false;
		float exposure_ev = 0.f;
		uint64_t seed = 0;
		std::string linear_output, metrics_output;
		std::vector<std::pair<std::string, std::string>> overrides;

		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];

			if (arg == "--help") {
				std::printf("Usage: xenon [--scene FILE] [--samples N] [--backend cpu|cuda|optix]\n"
							"             [--min-bounces N] [--max-bounces N] [--no-display]\n"
							"             [--output FILE] [--output-dir DIR] [--exposure EV]\n"
							"[--width W] [--height H] [--seed N] [--linear-output FILE.pfm] "
							"[--metrics FILE.json]\n"
							"Preview and PNG: exposure in stops (default 0), Reinhard tone "
							"mapping, sRGB.\n");

				return 0;
			} else if (arg == "--no-display") {
				no_display = true;
			} else if (arg == "--scene" || arg == "--samples" || arg == "--output" ||
					   arg == "--output-dir" || arg == "--min-bounces" || arg == "--max-bounces" ||
					   arg == "--backend" || arg == "--exposure" || arg == "--width" ||
					   arg == "--height" || arg == "--seed" || arg == "--linear-output" ||
					   arg == "--metrics") {
				if (i + 1 == argc || std::string(argv[i + 1]).starts_with("--"))
					throw std::runtime_error("Missing value for " + arg);
				std::string value = argv[++i];

				if (arg == "--scene") {
					if (!scene_path.empty())
						throw std::runtime_error("Scene specified more than once");
					scene_path = value;
				} else {
					overrides.emplace_back(arg, value);
				}
			} else if (!arg.empty() && arg[0] != '-' && scene_path.empty()) {
				scene_path = arg;
			} else {
				throw std::runtime_error("Unknown argument: " + arg);
			}
		}

		if (scene_path.empty())
			scene_path = "scenes/cornell_box/cornell_box.xenon";

		Scene scene;
		Camera camera;

		if (!load_scene(scene_path, scene, camera, config))
			return 1;

		for (const auto &[arg, value] : overrides) {
			if (arg == "--samples")
				config.samples = parse_int(value, arg);
			else if (arg == "--min-bounces")
				config.min_bounces = parse_int(value, arg);
			else if (arg == "--max-bounces")
				config.max_bounces = parse_int(value, arg);
			else if (arg == "--output")
				config.output = value;
			else if (arg == "--output-dir")
				output_dir = value;
			else if (arg == "--backend")
				config.backend = value;
			else if (arg == "--exposure")
				exposure_ev = parse_float(value, arg);
			else if (arg == "--width")
				config.width = parse_int(value, arg);
			else if (arg == "--height")
				config.height = parse_int(value, arg);
			else if (arg == "--linear-output") {
				if (value.empty())
					throw std::runtime_error("Output filename must not be empty");
				linear_output = value;
			} else if (arg == "--metrics") {
				if (value.empty())
					throw std::runtime_error("Output filename must not be empty");
				metrics_output = value;
			} else if (arg == "--seed") {
				auto [end, error] =
					std::from_chars(value.data(), value.data() + value.size(), seed);

				if (error != std::errc{} || end != value.data() + value.size())
					throw std::runtime_error("Invalid unsigned integer for --seed: " + value);
			}
		}

		if (config.width <= 0 || config.height <= 0 ||
			uint64_t(config.width) * config.height > 100000000)
			throw std::runtime_error("Invalid image dimensions");
		camera.set_aspect(float(config.width) / config.height);
		(void)DisplayTransform(exposure_ev); // Validate even when no preview is requested.
		if (config.samples <= 0)
			throw std::runtime_error("Samples must be positive");

		if (config.min_bounces < 0 || config.max_bounces <= 0 ||
			config.min_bounces > config.max_bounces)
			throw std::runtime_error(
				"Bounces must satisfy 0 <= min-bounces <= max-bounces, with max-bounces > 0");

		if (config.backend != "cpu" && config.backend != "cuda" && config.backend != "optix")
			throw std::runtime_error("Unknown backend '" + config.backend +
									 "' (expected cpu, cuda or optix)");
#ifndef XENON_HAS_OPTIX
		if (config.backend == "optix")
			throw std::runtime_error("OptiX backend is not available in this build");
#endif
#ifndef XENON_HAS_CUDA
		if (config.backend == "cuda")
			throw std::runtime_error("CUDA backend is not available in this build");
#else
		if (config.backend == "cuda" || config.backend == "optix") {
			int devices = 0;
			cudaError_t status = cudaGetDeviceCount(&devices);

			if (status != cudaSuccess || devices == 0)
				throw std::runtime_error(
					std::string("CUDA backend is unavailable: ") +
					(status == cudaSuccess ? "no devices" : cudaGetErrorString(status)));
		}
#endif
		if (config.output.empty())
			throw std::runtime_error("Output filename must not be empty");
		std::filesystem::path final_output = std::filesystem::path(output_dir) / config.output;

		if (!final_output.parent_path().empty())
			std::filesystem::create_directories(final_output.parent_path());

		metrics.scene = std::filesystem::absolute(scene_path).string();
		metrics.backend = config.backend;
		metrics.device = cpu_model();
		metrics.seed = seed;
		metrics.width = config.width;
		metrics.height = config.height;
		metrics.samples = config.samples;
		metrics.min_bounces = config.min_bounces;
		metrics.max_bounces = config.max_bounces;
		metrics.exposure = exposure_ev;
		metrics.output = final_output.string();
		metrics.load = config.load_seconds;
		metrics.acceleration = config.acceleration_seconds;

		if (!linear_output.empty())
			metrics.linear_output = (std::filesystem::path(output_dir) / linear_output).string();
		std::printf("[Backend] %s\n", config.backend == "optix"	 ? "OptiX RTX wavefront renderer"
									  : config.backend == "cuda" ? "CUDA GPU wavefront renderer"
																 : "CPU tiled renderer");
		std::printf("[Seed] %llu\n", static_cast<unsigned long long>(seed));
		std::printf("[Config] %dx%d, %d spp, bounces %d-%d, %s\n", config.width, config.height,
					config.samples, config.min_bounces, config.max_bounces,
					no_display ? "headless" : "preview");
		std::printf("[Display] Exposure %.2f EV, Reinhard tone mapping, sRGB\n", exposure_ev);

		std::unique_ptr<Window> win;

		if (!no_display) {
			win = std::make_unique<Window>(config.width, config.height,
										   "Xenon (" + scene_path + ")", exposure_ev);

			if (!win->valid())
				throw std::runtime_error(
					"Could not initialize preview; use --no-display for headless rendering");
		}
		std::vector<std::filesystem::path> outputs{
			std::filesystem::absolute(final_output).lexically_normal()};

		for (const auto &name : {linear_output, metrics_output}) {
			if (name.empty())
				continue;
			auto path = std::filesystem::absolute(std::filesystem::path(output_dir) / name)
							.lexically_normal();

			if (std::find(outputs.begin(), outputs.end(), path) != outputs.end())
				throw std::runtime_error(
					"PNG, linear output, and metrics must use different files");
			outputs.push_back(path);
		}
		auto setup_start = MeasurementClock::now();
		TripleSwapchain swapchain(config.width, config.height);
		std::unique_ptr<WavefrontRenderer> cpu_renderer;
#ifdef XENON_HAS_CUDA
		std::unique_ptr<CudaRenderer> cuda_renderer;

		if (config.backend == "cuda" || config.backend == "optix") {
			cudaDeviceProp device{};
			int device_index = 0;
			cudaGetDevice(&device_index);
			cudaGetDeviceProperties(&device, device_index);
			metrics.device = device.name;
			cudaMemGetInfo(&metrics.gpu_free_before, &metrics.gpu_total);
			cuda_renderer =
				std::make_unique<CudaRenderer>(config.width, config.height, config.max_bounces,
											   config.min_bounces, config.backend == "optix");
			cuda_renderer->set_seed(seed);
			metrics.setup = elapsed_seconds(setup_start);
			auto upload_start = MeasurementClock::now();
			cuda_renderer->upload_scene(scene);
			metrics.optix_build = cuda_renderer->acceleration_build_seconds();
			metrics.upload = std::max(0.0, elapsed_seconds(upload_start) - metrics.optix_build);
			cudaMemGetInfo(&metrics.gpu_free_after, &metrics.gpu_total);
		} else
#endif
		{
			cpu_renderer = std::make_unique<WavefrontRenderer>(config.width, config.height);
			cpu_renderer->set_bounces(config.min_bounces, config.max_bounces);
			cpu_renderer->set_seed(seed);
			metrics.setup = elapsed_seconds(setup_start);
		}

		auto render_start = MeasurementClock::now();
		int current_spp = 0;
		std::atomic<bool> exit_flag(false), done(false);
		std::exception_ptr render_error;
		auto render = [&]() {
			try {
				while (current_spp < config.samples && !exit_flag.load()) {
#ifdef XENON_HAS_CUDA
					if (cuda_renderer) {
						cuda_renderer->render_frame(scene, camera, swapchain, !no_display);
						current_spp = cuda_renderer->spp();
					} else
#endif
					{
						cpu_renderer->render_frame_tiled(scene, camera, swapchain);
						current_spp = cpu_renderer->get_spp();
					}

					if (current_spp % 10 == 0 || current_spp == config.samples)
						std::printf("Progress: %d/%d samples\n", current_spp, config.samples);
				}
			} catch (...) {
				render_error = std::current_exception();
			}
			done.store(true, std::memory_order_release);
		};

		if (no_display) {
			render();
		} else {
			std::thread render_thread(render);

			while (!done.load(std::memory_order_acquire) && !win->should_close()) {
				win->poll_events();
				win->display(swapchain);
				win->swap_buffers();
			}
			exit_flag.store(true);
			render_thread.join();
		}

		if (render_error)
			std::rethrow_exception(render_error);

		if (current_spp != config.samples) {
			std::fprintf(stderr, "[Cancelled] Render stopped at %d/%d samples\n", current_spp,
						 config.samples);

			return 2;
		}
		// Include all queued GPU work in rendering, and isolate final headless readback.
#ifdef XENON_HAS_CUDA
		if (cuda_renderer) {
			auto status = cudaDeviceSynchronize();

			if (status != cudaSuccess)
				throw std::runtime_error(cudaGetErrorString(status));
		}
#endif
		metrics.render = elapsed_seconds(render_start);
#ifdef XENON_HAS_CUDA
		if (cuda_renderer) {
			if (no_display)
				cuda_renderer->publish(swapchain);
			metrics.readback = cuda_renderer->readback_seconds();

			if (!no_display)
				metrics.render = std::max(0.0, metrics.render - metrics.readback);
		}
#endif
		auto output_start = MeasurementClock::now();
		metrics.inspect(swapchain.get_read_buffer());

		if (!metrics.linear_output.empty())
			save_linear_pfm(metrics.linear_output, config.width, config.height,
							swapchain.get_read_buffer());

		if (!save_image(final_output.string(), config.width, config.height,
						swapchain.get_read_buffer(), exposure_ev))
			return 1;
		metrics.output_time = elapsed_seconds(output_start);
		metrics.total = elapsed_seconds(total_start);

		if (!metrics_output.empty())
			metrics.save(std::filesystem::path(output_dir) / metrics_output);
		std::printf("[Timing] load=%.6f cpu_build=%.6f setup=%.6f upload=%.6f optix_build=%.6f "
					"render=%.6f readback=%.6f output=%.6f total=%.6f seconds\n",
					metrics.load, metrics.acceleration, metrics.setup, metrics.upload,
					metrics.optix_build, metrics.render, metrics.readback, metrics.output_time,
					metrics.total);
		std::printf("Total Render Time: %lld ms\n", static_cast<long long>(metrics.total * 1000));

		return 0;
	} catch (const std::exception &e) {
		std::fprintf(stderr, "[Error] %s\n", e.what());

		return 1;
	}
}
