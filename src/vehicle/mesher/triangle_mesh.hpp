#pragma once

#include <vector>

#include "core/eigen_types.hpp"

namespace bm {
namespace mesher {

using namespace core;


// Represents a 3D triangle mesh.
struct TriangleMesh
{
  TriangleMesh() = default;

  TriangleMesh(const std::vector<Vector3d>& vertices,
               const std::vector<Vector3i>& triangles)
      : vertices(vertices), triangles(triangles) {}

  std::vector<Vector3d> vertices;
  std::vector<Vector3i> triangles;
};


}
}
