#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace bm {
namespace core {

// Put Eigen vector types in our namespace.
typedef Eigen::Vector2f Vector2f;
typedef Eigen::Vector2d Vector2d;
typedef Eigen::Vector2i Vector2i;

typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::Vector3i Vector3i;

typedef Eigen::Vector4f Vector4f;
typedef Eigen::Vector4d Vector4d;
typedef Eigen::Vector4i Vector4i;

// 3D transformations.
typedef Eigen::AffineCompact3f Transform3f;
typedef Eigen::AffineCompact3d Transform3d;

// Bounding boxes.
typedef Eigen::AlignedBox2f Box2f;
typedef Eigen::AlignedBox2d Box2d;
typedef Eigen::AlignedBox2i Box2i;

}
}
