cmake_minimum_required(VERSION 3.20)

# ── CUDA support (optional) ───────────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
option(XENON_ENABLE_CUDA "Enable CUDA GPU backend" ON)

if(XENON_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  project(xenon LANGUAGES CXX CUDA)
  set(XENON_HAS_CUDA TRUE)
  message(STATUS "CUDA enabled — compiler: ${CMAKE_CUDA_COMPILER}")
else()
  project(xenon LANGUAGES CXX)
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

# ── Build type defaults ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -mavx2 -mfma -ffast-math -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0 -g -fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# ── Source groups ─────────────────────────────────────────────────────────────
set(GEOMETRY_SRC    src/geometry/blas.cpp src/geometry/tlas.cpp src/geometry/mesh.cpp src/geometry/prim_blas.cpp)
set(MATERIAL_SRC    src/material/bsdf.cpp src/material/material.cpp)
set(RENDER_SRC      src/render/wavefront.cpp src/render/cpu_tiled.cpp)
set(SCENE_SRC       src/scene/scene_file.cpp)
set(DISPLAY_SRC     src/display/window.cpp)

set(ALL_SRC
  ${GEOMETRY_SRC}
  ${MATERIAL_SRC}
  ${RENDER_SRC}
  ${SCENE_SRC}
  ${DISPLAY_SRC}
)

# ── CUDA sources (only when available) ────────────────────────────────────────
if(XENON_HAS_CUDA)
  set(CUDA_SRC
    src/cuda/gpu_scene.cu
    src/cuda/cuda_renderer.cu
  )
endif()

# ── Third-party ───────────────────────────────────────────────────────────────
add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write INTERFACE third_party)

# ── Main executable ───────────────────────────────────────────────────────────
if(XENON_HAS_CUDA)
  add_executable(xenon
    src/main.cpp
    ${ALL_SRC}
    ${CUDA_SRC}
  )
else()
  add_executable(xenon
    src/main.cpp
    ${ALL_SRC}
  )
endif()

target_include_directories(xenon PRIVATE src ${GLFW_INCLUDE_DIRS})
target_link_libraries(xenon PRIVATE
  ${GLFW_LIBRARIES}
  OpenGL::GL
  Threads::Threads
  stb_image_write
)
target_compile_options(xenon PRIVATE ${GLFW_CFLAGS_OTHER})

if(XENON_HAS_CUDA)
  target_compile_definitions(xenon PRIVATE XENON_HAS_CUDA=1)
  # Add CUDA toolkit include dir so CXX files can include <cuda_runtime.h>
  target_include_directories(xenon PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set_target_properties(xenon PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  # Auto-detect GPU architecture, fallback to Ada Lovelace (89)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(xenon PROPERTIES CUDA_ARCHITECTURES "native")
  endif()
  target_compile_options(xenon PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
  )
endif()

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()

function(add_xenon_test name src)
  add_executable(${name} ${src} ${ARGN})
  target_include_directories(${name} PRIVATE src third_party)
  target_link_libraries(${name} PRIVATE Threads::Threads stb_image_write)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_xenon_test(math_test     tests/math_test.cpp)
add_xenon_test(sampler_test  tests/sampler_test.cpp)
add_xenon_test(bsdf_test     tests/bsdf_test.cpp ${MATERIAL_SRC})
add_xenon_test(wavefront_test tests/wavefront_test.cpp)
