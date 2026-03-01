#pragma once
// scene/scene_file.h — Loader for .xenon scene files

#include "scene/scene.h"
#include "camera/camera.h"
#include <string>

namespace xn {

struct SceneConfig {
    int width = 800;
    int height = 600;
    int samples = 64;
    int min_bounces = 3;
    int max_bounces = 8;
    std::string output = "output.png";
};

// Loads a .xenon file and populates the scene and camera
bool load_scene(const std::string& path, Scene& scene, Camera& camera, SceneConfig& config);

} // namespace xn
