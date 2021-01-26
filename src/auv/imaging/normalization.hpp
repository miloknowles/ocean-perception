#pragma once

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;


Image3f Normalize(const Image3f& bgr);


// Increase the dynamic range of an image with (image - vmin) / (vmax - vmin).
Image3f EnhanceContrast(const Image3f& bgr);


Image3f EnhanceContrastFactor(const Image3f& bgr);


Image3f WhiteBalanceSimple(const Image3f& bgr);


Image3f LinearToGamma(const Image3f& bgr_linear, float gamma_power = 0.4545f);
Image3f GammaToLinear(const Image3f& bgr_gamma, float gamma_power = 2.2f);


// Clip the image to the range [vmin, vmax], and then stretch to be [0, 1].
Image3f EnhanceContrastDerya(const Image3f& bgr, float vmin, float vmax);


// Normalizes the image so that its average pixel color is gray.
Image3f CorrectColorRatio(const Image3f& bgr);


// Normalizes the colors in an image based on local illuminant.
// The result is more grayish, with the global color cast removed.
Image3f NormalizeColorIlluminant(const Image3f bgr);


Image1f Sharpen(const Image1f& gray);


}
}
