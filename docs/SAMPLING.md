# Reproducible path sampling

CPU, CUDA and OptiX use the same PCG generator. Each path packs its 32-bit pixel index and 32-bit sample index into a 64-bit key, combines it with a mixed 64-bit user seed, and derives separately mixed PCG state and stream selectors. All user-seed bits participate, including the high bit that PCG's stream selector alone would discard. The implementation is shared in `src/camera/path_seed.h`; it uses the [public-domain SplitMix64 mixing function](https://prng.di.unimi.it/splitmix64.c).

Seeding every pixel on one PCG stream from `pixel + sample * 0xdeadbeef` produced visible spatial structure. This was a problem in path initialization, not a failure of PCG's long-stream marginal distribution. The [PCG author discusses correlated stream initialization](https://www.pcg-random.org/posts/critiquing-pcg-streams.html).

The investigation used 384×96 pixel positions within a 512-wide image, 8,192 samples per pixel and user seed 42. For the seventh random variate (light selection after camera, lens and medium sampling), adjacent horizontal pixel means had correlation 0.8995 under the old initialization and 0.00336 with mixed initialization. Its row-mean standard deviation fell from 0.004015 to 0.0001688, close to the independent-uniform expectation of 0.0001628. Across the first eight variates, the largest absolute neighbor correlation after mixing was 0.0129.

An actual CPU render of the glass/fog studio background, at 96×32 and 8,192 samples per pixel, confirmed the image effect. The ratio of vertical to horizontal second-difference RMS fell from 1.693 to 0.978. Mean luminance changed from 0.142863 to 0.143025 (0.113%); structured bands became ordinary sampling noise. These are measurements of the tested scene/seed, not a general certification of RNG independence.

`sampler_test` checks reproducibility, high-bit user seeds, distinct pixel/sample keys, and broad independent-uniform bounds on integrated spatial fields. Substituting the old initialization fails 20 checks. `cuda_bsdf_test` compares 8,192 integer variates exactly between CPU and device path generators. Renderer regression tests additionally compare the resulting transport images.

The larger diagnostic fields can be reproduced with `scripts/probe_path_sampling.cpp` (compile instructions are in its header). Run it with `0 8192 legacy.bin` and `1 8192 mixed.bin`; each output contains nine 96×384 row-major float32 arrays, suitable for NumPy loading and spatial correlation analysis.

This initialization change intentionally changes seeded images from earlier builds. Record the renderer revision and rebuild fresh reference measurements when comparing numerical results across revisions.
