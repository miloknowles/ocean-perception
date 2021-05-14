#pragma once

#include <opencv2/calib3d.hpp>

#include "vision_core/cv_types.hpp"

namespace bm {
namespace stereo {

using namespace core;

Image1f EstimateDisparity(const Image1b& il,
                          const Image1b& ir,
                          int num_disp = 64,
                          int block_size = 3);

}
}
