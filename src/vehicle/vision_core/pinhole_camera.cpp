#include "vision_core/pinhole_camera.hpp"

namespace bm {
namespace core {


PinholeCamera::PinholeCamera(double fx, double fy, double cx, double cy, double h, double w)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(h), width_(w)
{
  // Set up intrinsics matrices for later.
  K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
  K_inv_ = K_.inverse();
}


// Returns a PinholeCamera with intrinsics scaled based on a new image resolution.
// NOTE(milo): This only applies when an image is resized! Cropping will not change the focal
// length, but resizing will.
PinholeCamera PinholeCamera::Rescale(int new_height, int new_width) const
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


// Project 3D point in the camera's RDF frame.
Vector2d PinholeCamera::Project(const Vector3d& p_cam) const
{
  const Vector3d xy_h = K_ * p_cam;
  return xy_h.head<2>() / xy_h(2);
}


// Backproject a pixel location to a 3D point in the camera's RDF frame.
Vector3d PinholeCamera::Backproject(const Vector2d& xy, double depth) const
{
  const Vector3d xy_h(xy.x(), xy.y(), 1);
  return depth * K_inv_ * xy_h;
}


}
}
