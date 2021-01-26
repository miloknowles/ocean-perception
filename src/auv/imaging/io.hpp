#pragma once

#include "core/cv_types.hpp"

namespace bm {
namespace imaging {

using namespace core;

// Load the depth maps from Sea-thru paper.
Image1f LoadDepthTif(const std::string& filepath);

}
}
