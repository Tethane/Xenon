#pragma once
// render/wavefront.h — Wavefront rendering and progressive accumulation

#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"
#include "render/thread_pool.h"
#include "render/wavefront_state.h"

namespace xn {

enum RenderMode {
    RENDER_MODE_DEFAULT = 0,
    RENDER_MODE_NEE_ONLY,
    RENDER_MODE_BSDF_ONLY,
    RENDER_MODE_SELF_INTERSECTION,
    RENDER_MODE_COUNT
};

class WavefrontRenderer {
public:
    WavefrontRenderer(int width, int height);
    ~WavefrontRenderer();

    void render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);
    
    void reset_accumulation() { spp_ = 0; }
    int  get_spp() const { return spp_; }

    void set_bounces(int min_b, int max_b) { min_bounces_ = min_b; max_bounces_ = max_b; }
    void set_render_mode(RenderMode mode) { render_mode_ = mode; reset_accumulation(); }

private:
    int width_, height_;
    int spp_ = 0;
    int min_bounces_ = 3;
    int max_bounces_ = 8;
    RenderMode render_mode_ = RENDER_MODE_DEFAULT;
    std::vector<Vec3> accumulation_buffer_;
    std::unique_ptr<ThreadPool> pool_;

    struct Tile {
        int x0, y0, x1, y1;
    };
    std::vector<Tile> tiles_;

    void render_tile(const Tile& tile, const Scene& scene, const Camera& camera);
};

} // namespace xn
