#pragma once
// render/wavefront.h — Wavefront rendering and progressive accumulation

#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"
#include "render/thread_pool.h"
#include "render/wavefront_state.h"

namespace xn {

class WavefrontRenderer {
public:
    WavefrontRenderer(int width, int height);
    ~WavefrontRenderer();

    void render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);
    
    void reset_accumulation() { spp_ = 0; }
    int  get_spp() const { return spp_; }

    void set_bounces(int min_b, int max_b) { min_bounces_ = min_b; max_bounces_ = max_b; }

private:
    int width_, height_;
    int spp_ = 0;
    int min_bounces_ = 3;
    int max_bounces_ = 8;
    std::vector<Vec3> accumulation_buffer_;
    std::unique_ptr<ThreadPool> pool_;

    struct Tile {
        int x0, y0, x1, y1;
    };
    std::vector<Tile> tiles_;

    void render_tile(const Tile& tile, const Scene& scene, const Camera& camera);
};

} // namespace xn
