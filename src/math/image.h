#pragma once
// math/image.h - Quantitative Noise Measurement for Testing Render Quality

#include "vec3.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace xn {

struct NoiseStats
{
    // Squared normalized residual metrics
    float meanSquaredScore = 0.0f;   // average of residual^2 / (blur^2 + eps)
    float rmsScore = 0.0f;           // sqrt(meanSquaredScore)

    // Absolute normalized residual metrics
    float meanAbsolute = 0.0f;
    float medianAbsolute = 0.0f;
    float p95Absolute = 0.0f;
    float maxAbsolute = 0.0f;

    // Diagnostics
    float avgLuminance = 0.0f;
    float edgeThreshold = 0.0f;
    float epsilon = 0.0f;
    int validPixelCount = 0;
    int totalInteriorPixelCount = 0;
    float validPixelRatio = 0.0f;
};

class ImageNoiseEstimator
{
public:
    static NoiseStats computeResidualNoiseStats(
        const float* framebuffer,
        int width,
        int height)
    {
        NoiseStats result;

        if (!framebuffer || width <= 2 || height <= 2)
            return result;

        const int pixelCount = width * height;
        result.totalInteriorPixelCount = (width - 2) * (height - 2);

        std::vector<float> luminance(pixelCount);
        std::vector<float> blurred(pixelCount);

        computeLuminance(framebuffer, luminance.data(), pixelCount);
        result.avgLuminance = computeAverage(luminance.data(), pixelCount);

        blur3x3(luminance.data(), blurred.data(), width, height);

        result.edgeThreshold = std::max(1e-6f, result.avgLuminance * 0.1f);
        result.epsilon = std::max(1e-8f, result.avgLuminance * result.avgLuminance * 1e-4f);

        std::vector<float> absoluteResiduals;
        absoluteResiduals.reserve(result.totalInteriorPixelCount);

        double sumSquared = 0.0;
        double sumAbsolute = 0.0;
        int count = 0;

        for (int y = 1; y < height - 1; ++y)
        {
            for (int x = 1; x < width - 1; ++x)
            {
                const int i = y * width + x;

                const float gx =
                    0.5f * (luminance[y * width + (x + 1)] -
                            luminance[y * width + (x - 1)]);

                const float gy =
                    0.5f * (luminance[(y + 1) * width + x] -
                            luminance[(y - 1) * width + x]);

                const float grad = std::sqrt(gx * gx + gy * gy);

                // Reject strong edges so scene detail is less likely
                // to be counted as noise.
                if (grad >= result.edgeThreshold)
                    continue;

                const float residual = luminance[i] - blurred[i];
                const float denom = blurred[i] * blurred[i] + result.epsilon;

                const float normalizedSquared = (residual * residual) / denom;
                const float normalizedAbsolute = std::fabs(residual) / std::sqrt(denom);

                sumSquared += normalizedSquared;
                sumAbsolute += normalizedAbsolute;
                absoluteResiduals.push_back(normalizedAbsolute);

                if (normalizedAbsolute > result.maxAbsolute)
                    result.maxAbsolute = normalizedAbsolute;

                ++count;
            }
        }

        result.validPixelCount = count;
        result.validPixelRatio =
            (result.totalInteriorPixelCount > 0)
                ? static_cast<float>(count) / static_cast<float>(result.totalInteriorPixelCount)
                : 0.0f;

        if (count == 0)
            return result;

        result.meanSquaredScore = static_cast<float>(sumSquared / static_cast<double>(count));
        result.rmsScore = std::sqrt(result.meanSquaredScore);
        result.meanAbsolute = static_cast<float>(sumAbsolute / static_cast<double>(count));

        result.medianAbsolute = computePercentileInPlace(absoluteResiduals, 0.50f);
        result.p95Absolute = computePercentileInPlace(absoluteResiduals, 0.95f);

        return result;
    }

private:
    static void computeLuminance(const float* framebuffer, float* lum, int pixelCount)
    {
        for (int i = 0; i < pixelCount; ++i)
        {
            const float r = framebuffer[i * 3 + 0];
            const float g = framebuffer[i * 3 + 1];
            const float b = framebuffer[i * 3 + 2];

            lum[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
    }

    static float computeAverage(const float* data, int n)
    {
        if (n <= 0)
            return 0.0f;

        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += data[i];

        return static_cast<float>(sum / static_cast<double>(n));
    }

    static inline int clampInt(int v, int lo, int hi)
    {
        return (v < lo) ? lo : (v > hi ? hi : v);
    }

    static void blur3x3(const float* src, float* dst, int width, int height)
    {
        static constexpr float kernel[3][3] =
        {
            {1.0f, 2.0f, 1.0f},
            {2.0f, 4.0f, 2.0f},
            {1.0f, 2.0f, 1.0f}
        };

        static constexpr float norm = 16.0f;

        auto sample = [&](int x, int y) -> float
        {
            x = clampInt(x, 0, width - 1);
            y = clampInt(y, 0, height - 1);
            return src[y * width + x];
        };

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        sum += kernel[ky + 1][kx + 1] * sample(x + kx, y + ky);
                    }
                }

                dst[y * width + x] = sum / norm;
            }
        }
    }

    static float computePercentileInPlace(std::vector<float>& values, float p)
    {
        if (values.empty())
            return 0.0f;

        p = std::clamp(p, 0.0f, 1.0f);

        const size_t n = values.size();
        const size_t k = static_cast<size_t>(p * static_cast<float>(n - 1));

        std::nth_element(values.begin(), values.begin() + k, values.end());
        return values[k];
    }
};

} // namespace xn