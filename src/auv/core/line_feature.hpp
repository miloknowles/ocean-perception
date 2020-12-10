#pragma once

#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "core/eigen_types.hpp"

namespace bm {
namespace core {

namespace ld = cv::line_descriptor;

struct LineFeature2D {
  // Need a default constructor for allocating empty vectors.
  LineFeature2D() = default;

  LineFeature2D(const Vector2d& ps, const Vector2d& pe)
      : p_start(ps), p_end(pe)
  {
    ComputeCrossProduct();
  }

  LineFeature2D(const ld::KeyLine& kl)
      : p_start(kl.startPointX, kl.startPointY),
      p_end(kl.endPointX, kl.endPointY)
  {
    ComputeCrossProduct();
  }

  Vector2d p_start = Vector2d::Zero();
  Vector2d p_end = Vector2d::Zero();
  Vector3d cross = Vector3d::Zero();

 private:
  void ComputeCrossProduct()
  {
    // Precompute the cross product used for projection error.
    // https://github.com/rubengooj/stvo-pl/blob/master/src/stereoFrame.cpp
    Vector3d ps_h, pe_h;
    ps_h << p_start, 1.0;
    pe_h << p_end, 1.0;
    cross = ps_h.cross(pe_h);
    cross /= std::sqrt(cross(0)*cross(0) + cross(1)*cross(1));
  }
};


struct LineFeature3D {
  // Need a default constructor for allocating empty vectors.
  LineFeature3D() = default;

  LineFeature3D(const Vector3d& Ps, const Vector3d& Pe) : P_start(Ps), P_end(Pe) {}

  Vector3d P_start = Vector3d::Zero();
  Vector3d P_end = Vector3d::Zero();
};


}
}
