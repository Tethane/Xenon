#include "optix/optix_backend.h"
#include "optix/optix_params.h"
#include "scene/scene.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "optix_ptx.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <mutex>
#include <vector>

namespace xn {
namespace {
void check(OptixResult result, const char *call) {
	if (result != OPTIX_SUCCESS)
		throw std::runtime_error(std::string("OptiX ") + call + ": " + optixGetErrorName(result));
}

void cuda_check(cudaError_t result) {
	if (result != cudaSuccess)
		throw std::runtime_error(std::string("OptiX CUDA: ") + cudaGetErrorString(result));
}

#define OPTIX_CHECK(call) check((call), #call)

struct DeviceBuffer {
	CUdeviceptr ptr = 0;
	size_t bytes = 0;
	DeviceBuffer() = default;

	explicit DeviceBuffer(size_t size) : bytes(size) {
		if (size)
			cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr), size));
	}

	~DeviceBuffer() {
		if (ptr)
			cudaFree(reinterpret_cast<void *>(ptr));
	}

	DeviceBuffer(const DeviceBuffer &) = delete;
	DeviceBuffer &operator=(const DeviceBuffer &) = delete;

	DeviceBuffer(DeviceBuffer &&other) noexcept
		: ptr(std::exchange(other.ptr, 0)), bytes(std::exchange(other.bytes, 0)) {}

	DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
		std::swap(ptr, other.ptr);
		std::swap(bytes, other.bytes);

		return *this;
	}

	void upload(const void *data, size_t size) {
		cuda_check(cudaMemcpy(reinterpret_cast<void *>(ptr), data, size, cudaMemcpyHostToDevice));
	}
};

template <class T> DeviceBuffer upload(const std::vector<T> &data) {
	DeviceBuffer result(data.size() * sizeof(T));

	if (!data.empty())
		result.upload(data.data(), result.bytes);

	return result;
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
	char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void log_callback(unsigned int level, const char *tag, const char *message, void *) {
	if (level <= 2)
		std::fprintf(stderr, "[OptiX][%s] %s\n", tag, message);
}
} // namespace

struct OptixBackend::Impl {
	OptixDeviceContext context = nullptr;
	OptixModule module = nullptr;
	OptixPipeline pipeline = nullptr;
	std::vector<OptixProgramGroup> groups;
	OptixShaderBindingTable sbt{};
	DeviceBuffer records, parameters, plane_refs;

	struct CachedLaunch {
		OptixParams params;
		DeviceBuffer device;
	};

	std::vector<CachedLaunch> launches;
	std::vector<DeviceBuffer> acceleration;
	OptixTraversableHandle handle = 0;
	double seconds = 0;
	int num_planes = 0;
	bool cache_parameters = true;

	~Impl() {
		// Destruction is only reached after outstanding renderer work has finished.
		cudaDeviceSynchronize();

		if (pipeline)
			optixPipelineDestroy(pipeline);

		for (auto group : groups)
			optixProgramGroupDestroy(group);

		if (module)
			optixModuleDestroy(module);

		if (context)
			optixDeviceContextDestroy(context);
	}

	void initialize() {
		const char *uploads = std::getenv("XENON_OPTIX_UPLOAD_PARAMS");
		cache_parameters = !(uploads && std::strcmp(uploads, "1") == 0);
		cuda_check(cudaFree(nullptr));
		static std::once_flag initialized;
		std::call_once(initialized, [] { OPTIX_CHECK(optixInit()); });
		OptixDeviceContextOptions options{};
		options.logCallbackFunction = log_callback;
		options.logCallbackLevel = 2;
		OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context));
		OptixModuleCompileOptions module_options{};
		module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
		OptixPipelineCompileOptions pipeline_options{};
		pipeline_options.traversableGraphFlags =
			OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipeline_options.numPayloadValues = 3;
		pipeline_options.numAttributeValues = 2;
		pipeline_options.pipelineLaunchParamsVariableName = "params";
		pipeline_options.usesPrimitiveTypeFlags =
			OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
		char log[8192]{};
		size_t log_size = sizeof(log);
		auto result =
			optixModuleCreate(context, &module_options, &pipeline_options, xenon_optix_ptx,
							  sizeof(xenon_optix_ptx) - 1, log, &log_size, &module);

		if (result != OPTIX_SUCCESS)
			throw std::runtime_error(std::string("OptiX module compilation: ") + log);
		OptixProgramGroupDesc desc[4]{};
		desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		desc[0].raygen.module = module;
		desc[0].raygen.entryFunctionName = "__raygen__queries";
		desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		desc[1].miss.module = module;
		desc[1].miss.entryFunctionName = "__miss__empty";

		for (int i = 2; i < 4; ++i) {
			desc[i].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			desc[i].hitgroup.moduleCH = module;
			desc[i].hitgroup.entryFunctionNameCH = "__closesthit__surface";
			desc[i].hitgroup.moduleAH = module;
			desc[i].hitgroup.entryFunctionNameAH = "__anyhit__surface";
		}
		desc[3].hitgroup.moduleIS = module;
		desc[3].hitgroup.entryFunctionNameIS = "__intersection__analytic";
		OptixProgramGroupOptions pg_options{};
		groups.resize(4);
		log_size = sizeof(log);
		OPTIX_CHECK(
			optixProgramGroupCreate(context, desc, 4, &pg_options, log, &log_size, groups.data()));
		OptixPipelineLinkOptions link_options{};
		link_options.maxTraceDepth = 1;
		log_size = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(context, &pipeline_options, &link_options, groups.data(), 4,
										log, &log_size, &pipeline));
		OptixStackSizes stack{};

		for (auto group : groups)
			OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack, pipeline));
		unsigned int traversal = 0, state = 0, continuation = 0;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
		OPTIX_CHECK(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 2));
		std::vector<SbtRecord> host_records(4);

		for (int i = 0; i < 4; ++i)
			OPTIX_CHECK(optixSbtRecordPackHeader(groups[i], &host_records[i]));
		records = upload(host_records);
		sbt.raygenRecord = records.ptr;
		sbt.missRecordBase = records.ptr + sizeof(SbtRecord);
		sbt.missRecordStrideInBytes = sizeof(SbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = records.ptr + 2 * sizeof(SbtRecord);
		sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
		sbt.hitgroupRecordCount = 2;
		parameters = DeviceBuffer(sizeof(OptixParams));
	}

	OptixTraversableHandle build_as(const OptixBuildInput &input) {
		OptixAccelBuildOptions options{};
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		options.operation = OPTIX_BUILD_OPERATION_BUILD;
		OptixAccelBufferSizes sizes{};
		OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes));
		DeviceBuffer temporary(sizes.tempSizeInBytes), output(sizes.outputSizeInBytes),
			property(sizeof(uint64_t));
		OptixAccelEmitDesc emit{property.ptr, OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};
		OptixTraversableHandle result = 0;
		OPTIX_CHECK(optixAccelBuild(context, nullptr, &options, &input, 1, temporary.ptr,
									temporary.bytes, output.ptr, output.bytes, &result, &emit, 1));
		uint64_t compacted_size = 0;
		cuda_check(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(property.ptr),
							  sizeof(compacted_size), cudaMemcpyDeviceToHost));

		if (compacted_size < output.bytes) {
			DeviceBuffer compacted(compacted_size);
			OPTIX_CHECK(optixAccelCompact(context, nullptr, result, compacted.ptr, compacted.bytes,
										  &result));
			cuda_check(cudaDeviceSynchronize());
			acceleration.push_back(std::move(compacted));
		} else
			acceleration.push_back(std::move(output));

		return result;
	}

	void launch(OptixParams &params, int count) {
		if (count <= 0)
			return;
		params.handle = handle;
		params.planes = reinterpret_cast<OptixPlaneRef *>(plane_refs.ptr);
		params.num_planes = num_planes;
		CUdeviceptr launch_parameters = parameters.ptr;

		if (cache_parameters && params.mode < 2) {
			// Scene data and queue buffers are immutable between build() calls. The
			// two ping-pong ray queues and shadow pass need three parameter records,
			// which can be reused without a synchronous host upload on every bounce.
			auto it =
				std::find_if(launches.begin(), launches.end(), [&](const CachedLaunch &cached) {
					const auto &p = cached.params;

					return p.mode == params.mode && p.paths == params.paths &&
						   p.queue == params.queue && p.counters == params.counters &&
						   p.hits == params.hits && p.shadows == params.shadows &&
						   p.scene.num_pixels == params.scene.num_pixels;
				});

			if (it == launches.end()) {
				CachedLaunch fresh{params, DeviceBuffer(sizeof(params))};
				fresh.device.upload(&params, sizeof(params));
				launches.push_back(std::move(fresh));
				launch_parameters = launches.back().device.ptr;
			} else
				launch_parameters = it->device.ptr;
		} else
			parameters.upload(&params, sizeof(params));
		OPTIX_CHECK(
			optixLaunch(pipeline, nullptr, launch_parameters, sizeof(params), &sbt, count, 1, 1));
	}
};

OptixBackend::OptixBackend() : impl_(std::make_unique<Impl>()) {
	impl_->initialize();
}

OptixBackend::~OptixBackend() = default;

void OptixBackend::build(const Scene &scene, GpuSceneData gpu_scene) {
	auto start = std::chrono::steady_clock::now();
	impl_->acceleration.clear();
	impl_->handle = 0;
	impl_->launches.clear();
	std::vector<OptixTraversableHandle> meshes(scene.meshes.size());
	int triangle_offset = 0;

	for (size_t i = 0; i < scene.meshes.size(); ++i) {
		const auto &mesh = scene.meshes[i];

		if (mesh.num_triangles() == 0)
			continue;
		std::vector<float3> vertices(mesh.num_vertices());

		for (int v = 0; v < mesh.num_vertices(); ++v)
			vertices[v] = make_float3(mesh.vx[v], mesh.vy[v], mesh.vz[v]);
		auto vertex_buffer = upload(vertices);
		OptixBuildInput input{};
		input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		auto &triangles = input.triangleArray;
		triangles.vertexBuffers = &vertex_buffer.ptr;
		triangles.numVertices = vertices.size();
		triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangles.vertexStrideInBytes = sizeof(float3);
		triangles.indexBuffer = reinterpret_cast<CUdeviceptr>(gpu_scene.indices + triangle_offset);
		triangles.numIndexTriplets = mesh.num_triangles();
		triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangles.indexStrideInBytes = 3 * sizeof(int32_t);
		unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
		triangles.flags = &flags;
		triangles.numSbtRecords = 1;
		meshes[i] = impl_->build_as(input);
		triangle_offset += mesh.num_triangles() * 3;
	}
	std::vector<OptixInstance> instances;
	std::vector<OptixPlaneRef> planes;
	const auto &cpu_instances = scene.tlas.instances();

	if (cpu_instances.size() > 0xffffff)
		throw std::runtime_error("OptiX supports at most 2^24 instance IDs");

	for (size_t i = 0; i < cpu_instances.size(); ++i) {
		const auto &inst = cpu_instances[i];
		OptixInstance oi{};

		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 4; ++col)
				oi.transform[4 * row + col] = inst.xform.local_to_world.cols[col][row];
		oi.instanceId = i;
		oi.visibilityMask = 255;

		if (inst.geom.type == GeomHandle::MESH_BLAS) {
			for (size_t m = 0; m < scene.blas_list.size(); ++m)
				if (scene.blas_list[m].get() == inst.geom.mesh)
					oi.traversableHandle = meshes[m];
			oi.sbtOffset = 0;
		} else {
			const auto &prims = inst.geom.prim->group().prims;
			std::vector<OptixAabb> bounds(prims.size());
			bool finite = false;

			for (size_t p = 0; p < prims.size(); ++p) {
				if (std::holds_alternative<Plane>(prims[p])) {
					planes.push_back({int(i), int(p)});
					bounds[p] = {0, 0, 0, 0, 0, 0};
				} else {
					AABB box = std::visit([](const auto &primitive) { return primitive.aabb(); },
										  prims[p]);
					bounds[p] = {box.mn.x, box.mn.y, box.mn.z, box.mx.x, box.mx.y, box.mx.z};
					finite = true;
				}
			}

			if (finite) {
				auto aabbs = upload(bounds);
				OptixBuildInput input{};
				input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
				input.customPrimitiveArray.aabbBuffers = &aabbs.ptr;
				input.customPrimitiveArray.numPrimitives = bounds.size();
				input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
				unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
				input.customPrimitiveArray.flags = &flags;
				input.customPrimitiveArray.numSbtRecords = 1;
				oi.traversableHandle = impl_->build_as(input);
			}
			oi.sbtOffset = 1;
		}

		if (oi.traversableHandle)
			instances.push_back(oi);
	}

	if (!instances.empty()) {
		auto device_instances = upload(instances);
		OptixBuildInput input{};
		input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		input.instanceArray.instances = device_instances.ptr;
		input.instanceArray.numInstances = instances.size();
		impl_->handle = impl_->build_as(input);
	}
	impl_->plane_refs = upload(planes);
	impl_->num_planes = planes.size();
	cuda_check(cudaDeviceSynchronize());
	impl_->seconds =
		std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	std::printf(
		"[OptiX] Built %zu GAS/IAS buffers, %zu instances, %zu infinite planes: %.3f s, %.2f MB\n",
		impl_->acceleration.size(), instances.size(), planes.size(), impl_->seconds,
		acceleration_bytes() / 1048576.0);
}

void OptixBackend::intersect(GpuSceneData scene, GpuPathState *paths, const int *queue,
							 int *counters, GpuHitRecord *hits, int count) {
	OptixParams params{};
	params.scene = scene;
	params.paths = paths;
	params.queue = queue;
	params.counters = counters;
	params.hits = hits;
	impl_->launch(params, count);
}

void OptixBackend::shadow(GpuSceneData scene, GpuPathState *paths, const GpuShadowWork *work,
						  int *counters, int count) {
	OptixParams params{};
	params.scene = scene;
	params.paths = paths;
	params.shadows = work;
	params.counters = counters;
	params.mode = 1;
	impl_->launch(params, count);
}

void OptixBackend::query(GpuSceneData scene, const GpuRay *rays, GpuHitRecord *hits, bool *occluded,
						 int count) {
	OptixParams params{};
	params.scene = scene;
	params.rays = rays;
	params.hits = hits;
	params.occluded = occluded;
	params.mode = 2;
	impl_->launch(params, count);
}

double OptixBackend::build_seconds() const {
	return impl_->seconds;
}

size_t OptixBackend::acceleration_bytes() const {
	size_t bytes = 0;

	for (const auto &buffer : impl_->acceleration)
		bytes += buffer.bytes;

	return bytes;
}
} // namespace xn
