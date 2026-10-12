cmake_minimum_required(VERSION 3.20)

# ── CUDA support (optional) ───────────────────────────────────────────────────
project(xenon LANGUAGES CXX)
include(CheckLanguage)
option(XENON_ENABLE_CUDA "Enable CUDA GPU backend" ON)
if(XENON_ENABLE_CUDA)
	check_language(CUDA)
endif()

if(XENON_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	set(XENON_HAS_CUDA TRUE)
	message(STATUS "CUDA enabled — compiler: ${CMAKE_CUDA_COMPILER}")
else()
	set(XENON_HAS_CUDA FALSE)
	if(XENON_ENABLE_CUDA)
		message(STATUS "CUDA requested but no compiler found — building CPU-only")
	else()
		message(STATUS "CUDA disabled by user — building CPU-only")
	endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# NVIDIA publishes the minimal OptiX headers separately from its SDK. Pin the exact revision so CI
# and benchmark builds use identical compiler interfaces.
option(XENON_ENABLE_OPTIX "Enable NVIDIA OptiX traversal backend" ${XENON_HAS_CUDA})
set(OPTIX_INCLUDE_DIR
	""
	CACHE PATH "OptiX SDK include directory (optional)")
if(XENON_ENABLE_OPTIX)
	if(NOT XENON_HAS_CUDA)
		message(FATAL_ERROR "OptiX requires CUDA; enable CUDA or disable XENON_ENABLE_OPTIX")
	endif()
	if(NOT OPTIX_INCLUDE_DIR)
		include(FetchContent)
		FetchContent_Declare(
			optix_headers
			GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
			GIT_TAG fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd)
		FetchContent_MakeAvailable(optix_headers)
		set(OPTIX_INCLUDE_DIR "${optix_headers_SOURCE_DIR}/include")
	endif()
	if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
		message(FATAL_ERROR "OPTIX_INCLUDE_DIR must contain optix.h")
	endif()
	set(XENON_HAS_OPTIX TRUE)
	message(STATUS "OptiX enabled: ${OPTIX_INCLUDE_DIR}")
	file(GLOB_RECURSE OPTIX_DEVICE_HEADERS CONFIGURE_DEPENDS src/*.cuh src/*.h src/*.inc)
	add_custom_command(
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx
		COMMAND
			${CMAKE_CUDA_COMPILER} --ptx --std=c++20 --gpu-architecture=compute_75
			--expt-relaxed-constexpr -O3 -lineinfo -I${CMAKE_CURRENT_SOURCE_DIR}/src
			-I${OPTIX_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/optix/optix_programs.cu -o
			${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx
		DEPENDS src/optix/optix_programs.cu ${OPTIX_DEVICE_HEADERS}
		VERBATIM)
	add_custom_command(
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h
		COMMAND
			${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx
			-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h -P
			${CMAKE_CURRENT_SOURCE_DIR}/src/optix/embed_ptx.cmake
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx src/optix/embed_ptx.cmake
		VERBATIM)
	add_custom_target(xenon_optix_ptx DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)
endif()

function(enable_xenon_optix target)
	if(XENON_HAS_OPTIX)
		target_sources(${target} PRIVATE src/optix/optix_backend.cu)
		target_include_directories(${target} PRIVATE ${OPTIX_INCLUDE_DIR}
													 ${CMAKE_CURRENT_BINARY_DIR})
		target_compile_definitions(${target} PRIVATE XENON_HAS_OPTIX=1)
		target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
		add_dependencies(${target} xenon_optix_ptx)
	endif()
endfunction()

# ── Build type defaults ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-mavx2;-mfma>")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>"
						$<$<COMPILE_LANGUAGE:CUDA>:-O3> -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_compile_options(-O0 -g $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
	add_link_options("$<HOST_LINK:-fsanitize=address>" "$<HOST_LINK:-fsanitize=undefined>")
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# ── Source groups ─────────────────────────────────────────────────────────────
set(GEOMETRY_SRC src/geometry/blas.cpp src/geometry/tlas.cpp src/geometry/mesh.cpp
				 src/geometry/prim_blas.cpp)
set(MATERIAL_SRC src/material/bsdf.cpp src/material/material.cpp src/material/texture.cpp)
set(RENDER_SRC src/render/wavefront.cpp src/render/cpu_tiled.cpp)
set(SCENE_SRC src/scene/scene_file.cpp)
set(DISPLAY_SRC src/display/window.cpp)

set(ALL_SRC ${GEOMETRY_SRC} ${MATERIAL_SRC} ${RENDER_SRC} ${SCENE_SRC} ${DISPLAY_SRC})

# ── CUDA sources (only when available) ────────────────────────────────────────
if(XENON_HAS_CUDA)
	set(CUDA_SRC src/cuda/gpu_scene.cu src/cuda/cuda_renderer.cu)
endif()

# ── Third-party ───────────────────────────────────────────────────────────────
add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write INTERFACE third_party)

# ── Main executable ───────────────────────────────────────────────────────────
if(XENON_HAS_CUDA)
	add_executable(xenon src/main.cpp ${ALL_SRC} ${CUDA_SRC})
else()
	add_executable(xenon src/main.cpp ${ALL_SRC})
endif()

target_include_directories(xenon PRIVATE src ${GLFW_INCLUDE_DIRS})
target_link_libraries(xenon PRIVATE ${GLFW_LIBRARIES} OpenGL::GL Threads::Threads stb_image_write)
target_compile_options(xenon PRIVATE ${GLFW_CFLAGS_OTHER})
enable_xenon_optix(xenon)

if(XENON_HAS_CUDA)
	target_compile_definitions(xenon PRIVATE XENON_HAS_CUDA=1)
	# Add CUDA toolkit include dir so CXX files can include <cuda_runtime.h>
	target_include_directories(xenon PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	set_target_properties(
		xenon
		PROPERTIES CUDA_STANDARD 20
				   CUDA_STANDARD_REQUIRED ON
				   CUDA_SEPARABLE_COMPILATION ON)
	# Auto-detect GPU architecture, fallback to Ada Lovelace (89)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set_target_properties(xenon PROPERTIES CUDA_ARCHITECTURES "native")
	endif()
	target_compile_options(xenon PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
										 -lineinfo>)
endif()

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()

function(add_xenon_test name src)
	add_executable(${name} ${src} ${ARGN})
	target_include_directories(${name} PRIVATE src third_party)
	target_link_libraries(${name} PRIVATE Threads::Threads stb_image_write)
	add_test(NAME ${name} COMMAND ${name})
endfunction()

add_xenon_test(math_test tests/math_test.cpp)
add_xenon_test(sampler_test tests/sampler_test.cpp)
add_xenon_test(camera_test tests/camera_test.cpp)
add_xenon_test(medium_test tests/medium_test.cpp)
add_xenon_test(environment_test tests/environment_test.cpp src/material/texture.cpp)
add_xenon_test(instance_test tests/instance_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC} ${SCENE_SRC})
add_xenon_test(bsdf_test tests/bsdf_test.cpp ${MATERIAL_SRC})
add_xenon_test(energy_test tests/energy_test.cpp ${MATERIAL_SRC})
add_xenon_test(wavefront_test tests/wavefront_test.cpp)

add_xenon_test(swapchain_test tests/swapchain_test.cpp)
add_xenon_test(display_test tests/display_test.cpp)
option(XENON_TEST_DISPLAY "Run OpenGL preview tests requiring a display server" OFF)
if(XENON_TEST_DISPLAY)
	add_xenon_test(window_test tests/display_test.cpp ${DISPLAY_SRC})
	target_compile_definitions(window_test PRIVATE XENON_TEST_WINDOW=1)
	target_include_directories(window_test PRIVATE ${GLFW_INCLUDE_DIRS})
	target_link_libraries(window_test PRIVATE ${GLFW_LIBRARIES} OpenGL::GL)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME execution_cpu
		 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/execution_test.py
				 $<TARGET_FILE:xenon> cpu $<BOOL:${XENON_HAS_CUDA}>)
add_test(NAME measurement_cpu
		 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/measurement_test.py
				 $<TARGET_FILE:xenon> cpu)
add_test(NAME transport_cpu
		 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/transport_test.py
				 $<TARGET_FILE:xenon> cpu)
option(XENON_TEST_CUDA "Run execution tests requiring a CUDA device" OFF)
if(XENON_TEST_CUDA AND XENON_HAS_CUDA)
	add_test(NAME execution_cuda
			 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/execution_test.py
					 $<TARGET_FILE:xenon> cuda 1)
	add_test(NAME measurement_cuda
			 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/measurement_test.py
					 $<TARGET_FILE:xenon> cuda)
	add_test(NAME transport_cuda
			 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/transport_test.py
					 $<TARGET_FILE:xenon> cuda)
	if(XENON_HAS_OPTIX)
		add_test(
			NAME measurement_optix
			COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/measurement_test.py
					$<TARGET_FILE:xenon> optix)
		add_test(NAME transport_optix
				 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/transport_test.py
						 $<TARGET_FILE:xenon> optix)
	endif()
endif()

add_xenon_test(geometry_test tests/geometry_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC})
if(XENON_HAS_CUDA)
	add_xenon_test(cuda_geometry_test tests/cuda_geometry_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   src/cuda/gpu_scene.cu)
	set_target_properties(cuda_geometry_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_geometry_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_geometry_test PROPERTIES DISABLED TRUE)
	endif()
endif()
if(XENON_HAS_CUDA)
	add_xenon_test(cuda_bsdf_test tests/cuda_bsdf_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   src/cuda/gpu_scene.cu)
	set_target_properties(cuda_bsdf_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_bsdf_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_bsdf_test PROPERTIES DISABLED TRUE)
	endif()
endif()

add_xenon_test(render_test tests/render_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC} ${RENDER_SRC})
add_xenon_test(texture_test tests/texture_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC} ${SCENE_SRC}
			   ${RENDER_SRC})
if(XENON_HAS_CUDA)
	add_xenon_test(
		cuda_texture_test
		tests/cuda_texture_test.cu
		${GEOMETRY_SRC}
		${MATERIAL_SRC}
		${SCENE_SRC}
		${RENDER_SRC}
		${CUDA_SRC})
	enable_xenon_optix(cuda_texture_test)
	set_target_properties(cuda_texture_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_texture_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_texture_test PROPERTIES DISABLED TRUE)
	endif()
	add_xenon_test(cuda_render_test tests/cuda_render_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   ${RENDER_SRC} ${CUDA_SRC})
	enable_xenon_optix(cuda_render_test)
	set_target_properties(cuda_render_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_render_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_render_test PROPERTIES DISABLED TRUE)
	endif()
endif()

if(XENON_HAS_OPTIX)
	add_xenon_test(
		optix_texture_test
		tests/optix_texture_test.cu
		${GEOMETRY_SRC}
		${MATERIAL_SRC}
		${SCENE_SRC}
		${RENDER_SRC}
		${CUDA_SRC})
	enable_xenon_optix(optix_texture_test)
	set_target_properties(optix_texture_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(optix_texture_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(optix_texture_test PROPERTIES DISABLED TRUE)
	endif()
	add_xenon_test(optix_geometry_test tests/optix_geometry_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   src/cuda/gpu_scene.cu)
	enable_xenon_optix(optix_geometry_test)
	set_target_properties(optix_geometry_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(optix_geometry_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(optix_geometry_test PROPERTIES DISABLED TRUE)
	endif()
	add_xenon_test(optix_render_test tests/optix_render_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   ${RENDER_SRC} ${CUDA_SRC})
	enable_xenon_optix(optix_render_test)
	set_target_properties(optix_render_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(optix_render_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(optix_render_test PROPERTIES DISABLED TRUE)
	endif()
endif()

add_xenon_test(opacity_test tests/opacity_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC})
if(XENON_HAS_CUDA)
	add_xenon_test(cuda_opacity_test tests/cuda_opacity_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   src/cuda/gpu_scene.cu)
	set_target_properties(cuda_opacity_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_opacity_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_opacity_test PROPERTIES DISABLED TRUE)
	endif()
endif()
if(XENON_HAS_OPTIX)
	add_xenon_test(optix_opacity_test tests/optix_opacity_test.cu ${GEOMETRY_SRC} ${MATERIAL_SRC}
				   src/cuda/gpu_scene.cu)
	enable_xenon_optix(optix_opacity_test)
	set_target_properties(optix_opacity_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(optix_opacity_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(optix_opacity_test PROPERTIES DISABLED TRUE)
	endif()
endif()

add_xenon_test(light_sampling_test tests/light_sampling_test.cpp ${GEOMETRY_SRC} ${MATERIAL_SRC})
if(XENON_HAS_CUDA)
	add_xenon_test(cuda_light_sampling_test tests/cuda_light_sampling_test.cu ${GEOMETRY_SRC}
				   ${MATERIAL_SRC} src/cuda/gpu_scene.cu)
	set_target_properties(cuda_light_sampling_test PROPERTIES CUDA_STANDARD 20
															  CUDA_STANDARD_REQUIRED ON)
	target_compile_options(cuda_light_sampling_test
						   PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	if(NOT XENON_TEST_CUDA)
		set_tests_properties(cuda_light_sampling_test PROPERTIES DISABLED TRUE)
	endif()
endif()
add_test(NAME opacity_transport_cpu
		 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opacity_transport_test.py
				 $<TARGET_FILE:xenon> cpu)
if(XENON_TEST_CUDA AND XENON_HAS_CUDA)
	add_test(
		NAME opacity_transport_cuda
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opacity_transport_test.py
				$<TARGET_FILE:xenon> cuda)
	if(XENON_HAS_OPTIX)
		add_test(
			NAME opacity_transport_optix
			COMMAND
				${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/opacity_transport_test.py
				$<TARGET_FILE:xenon> optix)
	endif()
endif()

add_xenon_test(thread_pool_test tests/thread_pool_test.cpp)
set_tests_properties(thread_pool_test PROPERTIES TIMEOUT 60)
set_tests_properties(render_test PROPERTIES TIMEOUT 60)

add_test(NAME emission_sidedness_cpu
		 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/emission_sidedness_test.py
				 $<TARGET_FILE:xenon> cpu)
if(XENON_TEST_CUDA AND XENON_HAS_CUDA)
	add_test(
		NAME emission_sidedness_cuda
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/emission_sidedness_test.py
				$<TARGET_FILE:xenon> cuda)
	if(XENON_HAS_OPTIX)
		add_test(
			NAME emission_sidedness_optix
			COMMAND
				${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/emission_sidedness_test.py
				$<TARGET_FILE:xenon> optix)
	endif()
endif()
