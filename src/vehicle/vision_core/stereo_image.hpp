#pragma once

#include "core/macros.hpp"
#include "core/timestamp.hpp"
#include "core/uid.hpp"
#include "vision_core/cv_types.hpp"

namespace bm {
namespace core {


template <typename ImageT>
struct StereoImage final
{
  MACRO_DELETE_DEFAULT_CONSTRUCTOR(StereoImage)

  explicit StereoImage(timestamp_t timestamp,
                       uid_t camera_id,
                       const ImageT& l,
                       const ImageT& r)
    : timestamp(timestamp),
      camera_id(camera_id),
      left_image(l),
      right_image(r) {}

  timestamp_t timestamp;
  uid_t camera_id;
  ImageT left_image;
  ImageT right_image;
};

typedef StereoImage<Image1b> StereoImage1b;
typedef StereoImage<Image3b> StereoImage3b;

}
}
