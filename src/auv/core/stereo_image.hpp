#pragma once

#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace core {


struct StereoImage final
{
  StereoImage(timestamp_t timestamp, uid_t camera_id, const Image1b& l, const Image1b& r)
    : timestamp(timestamp), camera_id(camera_id), left_image(l), right_image(r) {}

  timestamp_t timestamp;
  uid_t camera_id;
  Image1b left_image;
  Image1b right_image;
};


}
}
