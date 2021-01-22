#pragma once

#include "core/cv_types.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

// Increase the dynamic range of an image with (image - vmin) / (vmax - vmin).
Image3f EnhanceContrast(const Image3f& bgr);


Image3f EnhanceContrastFactor(const Image3f& bgr);


Image3f WhiteBalanceSimple(const Image3f& bgr);


Image3f LinearToGamma(const Image3f& bgr_linear);


Image3f GammaToLinear(const Image3f& bgr_gamma);


}
}
