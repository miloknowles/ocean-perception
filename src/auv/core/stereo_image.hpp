#pragma once

#include "core/timestamp.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace core {


struct StereoImage final {
  StereoImage(timestamp_t timestamp, const Image1b& l, const Image1b& r)
    : timestamp(timestamp), left_image(l), right_image(r) {}

  timestamp_t timestamp;
  Image1b left_image;
  Image1b right_image;
};


}
}
