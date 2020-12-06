#pragma once

#include "eigen_types.hpp"

namespace bm {
namespace core {


class PinholeCamera final {
 public:
  PinholeCamera(float fx, float fy, float cx, float cy, float h, float w)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(h), width_(w) {
    // Set up intrinsics matrices for later.
    K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    K_inv_ = K_.inverse();
  }

  // Returns a PinholeCamera with intrinsics scaled based on a new image resolution.
  // NOTE(milo): This only applies when an image is resized! Cropping will not change the focal
  // length, but resizing will.
  PinholeCamera Rescale(int new_height, int new_width) const
  {
    const float height_sf = new_height / static_cast<float>(height_);
    const float width_sf = new_width / static_cast<float>(width_);
    return PinholeCamera(fx_ * width_sf,
                         fy_ * height_sf,
                         cx_ * width_sf,
                         cy_ * height_sf,
                         new_height,
                         new_width);
  }

  float fx() const { return fx_; }
  float fy() const { return fy_; }
  float cx() const { return cx_; }
  float cy() const { return cy_; }
  float Width() const { return width_; }
  float Height() const { return height_; }

  // Project 3D point in the camera's RDF frame.
  Vector2f Project(const Vector3f& p_cam) const
  {
    const Vector3f uv_h = K_ * p_cam;
    return uv_h.head<2>() / uv_h(2);
  }

  // Backproject a pixel location to a 3D point in the camera's RDF frame.
  Vector3f Backproject(const Vector2f& p_uv, float depth) const
  {
    const Vector3f p_uv_h(p_uv.x(), p_uv.y(), 1);
    return depth * K_inv_ * p_uv_h;
  }

 private:
  float fx_, fy_, cx_, cy_;

  // Nominal width and height for an image captured by this camera.
  int height_, width_;

  Matrix3f K_ = Matrix3f::Identity();
  Matrix3f K_inv_ = Matrix3f::Identity();
};

}
}
