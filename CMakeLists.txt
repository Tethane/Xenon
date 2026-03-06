cmake_minimum_required(VERSION 3.20)
project(xenon LANGUAGES CXX)

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
set(MATH_SRC)       # header-only
set(GEOMETRY_SRC    src/geometry/bvh.cpp src/geometry/mesh.cpp)
set(CAMERA_SRC)     # header-only
set(MATERIAL_SRC    src/material/bsdf.cpp src/material/material.cpp)
set(RENDER_SRC      src/render/wavefront.cpp src/render/integrator.cpp)
set(SCENE_SRC       src/scene/scene_file.cpp)
set(DISPLAY_SRC     src/display/window.cpp)

set(ALL_SRC
  ${GEOMETRY_SRC}
  ${MATERIAL_SRC}
  ${RENDER_SRC}
  ${SCENE_SRC}
  ${DISPLAY_SRC}
)

# ── Third-party ───────────────────────────────────────────────────────────────
add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write INTERFACE third_party)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(xenon src/main.cpp ${ALL_SRC})
target_include_directories(xenon PRIVATE src ${GLFW_INCLUDE_DIRS})
target_link_libraries(xenon PRIVATE
  ${GLFW_LIBRARIES}
  OpenGL::GL
  Threads::Threads
  stb_image_write
)
target_compile_options(xenon PRIVATE ${GLFW_CFLAGS_OTHER})

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()

function(add_xenon_test name src)
  add_executable(${name} ${src} ${ARGN})
  target_include_directories(${name} PRIVATE src third_party)
  target_link_libraries(${name} PRIVATE Threads::Threads stb_image_write)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_xenon_test(math_test     tests/math_test.cpp)
add_xenon_test(geometry_test tests/geometry_test.cpp ${GEOMETRY_SRC})
add_xenon_test(sampler_test  tests/sampler_test.cpp)
add_xenon_test(bsdf_test     tests/bsdf_test.cpp ${MATERIAL_SRC})
add_xenon_test(wavefront_test tests/wavefront_test.cpp)
