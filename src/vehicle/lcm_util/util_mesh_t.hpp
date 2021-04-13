#pragma once

#include "core/eigen_types.hpp"

#include "vehicle/mesh_t.hpp"
#include "vehicle/vector3_t.hpp"

namespace bm {

using namespace core;


void pack_mesh_t(const std::vector<Vector3d>& vertices,
                 const std::vector<Vector3i>& triangles,
                 vehicle::mesh_t& msg)
{
  msg.num_vertices = (int32_t)vertices.size();
  msg.num_triangles = (int32_t)triangles.size();

  msg.vertices.clear();
  msg.triangles.clear();

  for (size_t i = 0; i < vertices.size(); ++i) {
    vehicle::vector3_t pt;
    pt.x = vertices[i].x();
    pt.y = vertices[i].y();
    pt.z = vertices[i].z();
    msg.vertices.emplace_back(std::move(pt));
  }

  for (size_t i = 0; i < triangles.size(); ++i) {
    vehicle::mesh_triangle_t tri;
    tri.vertex_indices[0] = triangles[i].x();
    tri.vertex_indices[1] = triangles[i].y();
    tri.vertex_indices[2] = triangles[i].z();
    msg.triangles.emplace_back(std::move(tri));
  }
}


}
