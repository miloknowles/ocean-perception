#pragma once

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


struct LineFeature2D {
  LineFeature2D(const Vector2d& ps, const Vector2d& pe) : p_start(ps), p_end(pe)
  {
    // Precompute the cross product used for projection error.
    // https://github.com/rubengooj/stvo-pl/blob/master/src/stereoFrame.cpp
    Vector3d ps_h, pe_h;
    ps_h << p_start, 1.0;
    pe_h << p_end, 1.0;
    cross = ps_h.cross(pe_h);
    cross /= std::sqrt(cross(0)*cross(0) + cross(1)*cross(1));
  }

  Vector2d p_start;
  Vector2d p_end;
  Vector3d cross;
};


struct LineFeature3D {
  LineFeature3D(const Vector3d& Ps, const Vector3d& Pe) : P_start(Ps), P_end(Pe) {}
  Vector3d P_start;
  Vector3d P_end;
};


}
}
