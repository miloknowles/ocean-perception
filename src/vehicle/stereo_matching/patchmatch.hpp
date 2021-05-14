#pragma once

#include "vision_core/cv_types.hpp"

namespace bm {
namespace stereo {

using namespace core;


// Returns a binary mask where "1" indicates foreground and "0" indicates background.
void ForegroundTextureMask(const Image1b& gray,
                          Image1b& mask,
                          int ksize = 7,
                          double min_grad = 35.0,
                          int downsize = 2);


}
}
