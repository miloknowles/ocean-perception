#pragma once

#include "core/cv_types.hpp"
#include "core/image_util.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace imaging {

using namespace core;


Image3f CorrectAttenuationSimple(const Image3f& bgr,
                                 const Image1f& range,
                                 const Vector3f& beta_D);


Image3f EstimateIlluminantGaussian(const Image3f& bgr,
                           const Image1f& range,
                           int ksizeX,
                           int ksizeY,
                           double sigmaX,
                           double sigmaY);


Image3f EstimateIlluminantGuided(const Image3f& bgr,
                                 const Image1f& range,
                                 int r,
                                 double eps,
                                 int s);


Image3f EnhanceUnderwater(const Image3f& bgr,
                          const Image1f& range,
                          float dark_percentile,
                          int backscatter_num_px,
                          int backscatter_opt_iters,
                          float brightness_boost);

}
}
