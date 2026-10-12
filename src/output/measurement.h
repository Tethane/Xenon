#pragma once

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

namespace xn {
using MeasurementClock = std::chrono::steady_clock;

inline double elapsed_seconds(MeasurementClock::time_point start) {
	return std::chrono::duration<double>(MeasurementClock::now() - start).count();
}

inline void prepare_output(const std::filesystem::path &path) {
	if (path.empty())
		throw std::runtime_error("Output filename must not be empty");

	if (!path.parent_path().empty())
		std::filesystem::create_directories(path.parent_path());
}

inline void save_linear_pfm(const std::filesystem::path &path, int width, int height,
							const float *rgb) {
	prepare_output(path);
	std::ofstream out(path, std::ios::binary);

	if (!out)
		throw std::runtime_error("Cannot open linear output: " + path.string());
	out << "PF\n"
		<< width << ' ' << height << '\n'
		<< (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");

	for (int y = height - 1; y >= 0; --y)
		out.write(reinterpret_cast<const char *>(rgb + size_t(y) * width * 3),
				  size_t(width) * 3 * sizeof(float));
	out.close();

	if (!out)
		throw std::runtime_error("Failed to write linear output: " + path.string());
}

inline std::string json_string(const std::string &value) {
	std::ostringstream out;
	out << '"';

	for (unsigned char c : value) {
		switch (c) {
			case '"':
				out << "\\\"";
				break;
			case '\\':
				out << "\\\\";
				break;
			case '\n':
				out << "\\n";
				break;
			case '\r':
				out << "\\r";
				break;
			case '\t':
				out << "\\t";
				break;
			default:
				if (c < 32)
					out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
						<< std::dec;
				else
					out << c;
		}
	}
	out << '"';

	return out.str();
}

inline std::string cpu_model() {
	std::ifstream in("/proc/cpuinfo");
	std::string line;

	while (std::getline(in, line)) {
		if (line.starts_with("model name")) {
			auto colon = line.find(':');

			return colon == std::string::npos ? line : line.substr(colon + 2);
		}
	}

	return "unknown";
}

struct RenderMeasurement {
	std::string scene, backend, device, output, linear_output;
	uint64_t seed = 0;
	int width = 0, height = 0, samples = 0, min_bounces = 0, max_bounces = 0;
	size_t gpu_free_before = 0, gpu_free_after = 0, gpu_total = 0;
	double load = 0, acceleration = 0, setup = 0, upload = 0, optix_build = 0, render = 0,
		   readback = 0, output_time = 0, total = 0;
	double mean_luminance = 0, max_radiance = 0;
	float exposure = 0;

	void inspect(const float *rgb) {
		double luminance = 0;

		for (size_t i = 0; i < size_t(width) * height; ++i) {
			for (int c = 0; c < 3; ++c) {
				float v = rgb[3 * i + c];

				if (!std::isfinite(v) || v < 0)
					throw std::runtime_error("Invalid linear radiance in final image");
				max_radiance = std::max(max_radiance, double(v));
			}
			luminance += 0.2126 * rgb[3 * i] + 0.7152 * rgb[3 * i + 1] + 0.0722 * rgb[3 * i + 2];
		}
		mean_luminance = luminance / (double(width) * height);
	}

	void save(const std::filesystem::path &path) const {
		prepare_output(path);
		std::ofstream out(path);

		if (!out)
			throw std::runtime_error("Cannot open metrics output: " + path.string());
		rusage usage{};
		getrusage(RUSAGE_SELF, &usage);
		out << std::setprecision(12) << "{\n  \"schema_version\": 1,\n"
			<< "  \"scene\": " << json_string(scene) << ",\n"
			<< "  \"backend\": " << json_string(backend) << ",\n"
			<< "  \"device\": " << json_string(device) << ",\n"
			<< "  \"cpu\": " << json_string(cpu_model()) << ",\n"
			<< "  \"compiler\": " << json_string(__VERSION__) << ",\n"
#ifdef NDEBUG
			<< "  \"build\": \"Release\",\n"
#else
			<< "  \"build\": \"Debug\",\n"
#endif
			<< "  \"gpu_host_counters\": "
			<< ((std::getenv("XENON_GPU_HOST_COUNTERS") &&
				 std::string(std::getenv("XENON_GPU_HOST_COUNTERS")) == "1")
					? "true"
					: "false")
			<< ",\n"
			<< "  \"optix_upload_params\": "
			<< ((std::getenv("XENON_OPTIX_UPLOAD_PARAMS") &&
				 std::string(std::getenv("XENON_OPTIX_UPLOAD_PARAMS")) == "1")
					? "true"
					: "false")
			<< ",\n"
			<< "  \"seed\": " << seed << ",\n"
			<< "  \"width\": " << width << ", \"height\": " << height
			<< ", \"samples\": " << samples << ",\n"
			<< "  \"min_bounces\": " << min_bounces << ", \"max_bounces\": " << max_bounces << ",\n"
			<< "  \"exposure_ev\": " << exposure << ",\n"
			<< "  \"display_transform\": \"exposure/reinhard/srgb\",\n"
			<< "  \"png\": " << json_string(output) << ",\n"
			<< "  \"linear_pfm\": " << json_string(linear_output) << ",\n"
			<< "  \"mean_luminance\": " << mean_luminance << ", \"max_radiance\": " << max_radiance
			<< ",\n"
			<< "  \"peak_host_rss_bytes\": " << uint64_t(usage.ru_maxrss) * 1024 << ",\n"
			<< "  \"gpu_free_before_bytes\": " << gpu_free_before << ",\n"
			<< "  \"gpu_free_after_bytes\": " << gpu_free_after << ",\n"
			<< "  \"gpu_total_bytes\": " << gpu_total << ",\n"
			<< "  \"seconds\": {\n"
			<< "    \"load\": " << load << ", \"cpu_acceleration\": " << acceleration << ",\n"
			<< "    \"renderer_setup\": " << setup << ", \"upload\": " << upload
			<< ", \"optix_build\": " << optix_build << ",\n"
			<< "    \"render\": " << render << ", \"readback\": " << readback << ",\n"
			<< "    \"output\": " << output_time << ", \"total\": " << total << "\n  }\n}\n";
		out.close();

		if (!out)
			throw std::runtime_error("Failed to write metrics output: " + path.string());
	}
};
} // namespace xn
