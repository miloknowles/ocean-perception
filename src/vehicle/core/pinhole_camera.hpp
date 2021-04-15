#pragma once

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


class PinholeCamera final {
 public:
  PinholeCamera() = default;
  PinholeCamera(double fx, double fy, double cx, double cy, double h, double w);

  // Returns a PinholeCamera with intrinsics scaled based on a new image resolution.
  // NOTE(milo): This only applies when an image is resized! Cropping will not change the focal
  // length, but resizing will.
  PinholeCamera Rescale(int new_height, int new_width) const;

  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double Width() const { return width_; }
  double Height() const { return height_; }
  Matrix3d K() const { return K_; }
  Matrix3d Kinv() const { return K_inv_; }

  // Project 3D point in the camera's RDF frame.
  Vector2d Project(const Vector3d& p_cam) const;

  // Backproject a pixel location to a 3D point in the camera's RDF frame.
  Vector3d Backproject(const Vector2d& xy, double depth) const;

 private:
  double fx_, fy_, cx_, cy_;

  // Nominal width and height for an image captured by this camera.
  int height_, width_;

  Matrix3d K_ = Matrix3d::Identity();
  Matrix3d K_inv_ = Matrix3d::Identity();
};

}
}
