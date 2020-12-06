#pragma once

#include "eigen_types.hpp"

namespace bm {
namespace core {


class PinholeCamera final {
 public:
  PinholeCamera(double fx, double fy, double cx, double cy, double h, double w)
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
    const double height_sf = new_height / static_cast<double>(height_);
    const double width_sf = new_width / static_cast<double>(width_);
    return PinholeCamera(fx_ * width_sf,
                         fy_ * height_sf,
                         cx_ * width_sf,
                         cy_ * height_sf,
                         new_height,
                         new_width);
  }

  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double Width() const { return width_; }
  double Height() const { return height_; }

  // Project 3D point in the camera's RDF frame.
  Vector2d Project(const Vector3d& p_cam) const
  {
    const Vector3d uv_h = K_ * p_cam;
    return uv_h.head<2>() / uv_h(2);
  }

  // Backproject a pixel location to a 3D point in the camera's RDF frame.
  Vector3d Backproject(const Vector2d& p_uv, double depth) const
  {
    const Vector3d p_uv_h(p_uv.x(), p_uv.y(), 1);
    return depth * K_inv_ * p_uv_h;
  }


 private:
  double fx_, fy_, cx_, cy_;

  // Nominal width and height for an image captured by this camera.
  int height_, width_;

  Matrix3d K_ = Matrix3d::Identity();
  Matrix3d K_inv_ = Matrix3d::Identity();
};

}
}
