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

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

typedef Eigen::Matrix<float, 12, 1> Vector12f;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<float, 12, 12> Matrix12f;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;

typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 16, 1> Vector16d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 16, 16> Matrix16d;

// Put Eigen matrix types in our namespace.
typedef Eigen::Matrix<float, 1, 1> Matrix1f;
typedef Eigen::Matrix3f Matrix3f;
typedef Eigen::Matrix4f Matrix4f;

typedef Eigen::Matrix<double, 1, 1> Matrix1d;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Matrix4d Matrix4d;

// 3D transformations.
typedef Eigen::AffineCompact3f Transform3f;
typedef Eigen::AffineCompact3d Transform3d;
typedef Eigen::AngleAxisf AngleAxisf;
typedef Eigen::AngleAxisd AngleAxisd;
typedef Eigen::Quaternionf Quaternionf;
typedef Eigen::Quaterniond Quaterniond;

// Bounding boxes.
typedef Eigen::AlignedBox2f Box2f;
typedef Eigen::AlignedBox2d Box2d;
typedef Eigen::AlignedBox2i Box2i;

}
}
