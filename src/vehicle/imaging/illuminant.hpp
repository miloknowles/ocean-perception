#pragma once

#include "core/cv_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

Image3f EstimateIlluminantGaussian(const Image3f& bgr,
                                  int ksizeX,
                                  int ksizeY,
                                  double sigmaX,
                                  double sigmaY);


Image3f EstimateIlluminantRangeGuided(const Image3f& bgr,
                                      const Image1f& range,
                                      int r,
                                      double eps,
                                      int s);

}
}
