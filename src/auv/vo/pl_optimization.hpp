#pragma once

#include <vector>

#include "core/eigen_types.hpp"
#include "core/stereo_camera.hpp"
#include "core/line_feature.hpp"

namespace bm {
namespace vo {

using namespace core;


int OptimizeOdometryIterativeL(const std::vector<Vector3d>& P0_list,
                            const std::vector<Vector2d>& p1_obs_list,
                            const std::vector<double>& p1_sigma_list,
                            const std::vector<LineFeature3D>& L0_list,
                            const std::vector<LineFeature2D>& l1_obs_list,
                            const std::vector<double>& l1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_10,
                            Matrix6d& C_10,
                            double& error,
                            std::vector<int>& point_inlier_indices,
                            std::vector<int>& line_inlier_indices,
                            int max_iters,
                            double min_error,
                            double min_error_delta,
                            double max_error_stdevs);


int OptimizeOdometryLML(const std::vector<Vector3d>& P0_list,
                                    const std::vector<Vector2d>& p1_obs_list,
                                    const std::vector<double>& p1_sigma_list,
                                    const std::vector<LineFeature3D>& L0_list,
                                    const std::vector<LineFeature2D>& l1_obs_list,
                                    const std::vector<double>& l1_sigma_list,
                                    const StereoCamera& stereo_cam,
                                    Matrix4d& T_10,
                                    Matrix6d& C_10,
                                    double& error,
                                    int max_iters,
                                    double min_error,
                                    double min_error_delta);



/**
 * Linearize line endpoint projection on the SE3 manifold at the current pose.
 *
 * @param L0_list : The 3D endpoint locations in the Camera_0 frame.
 * @param l1_obs_list : The 2D observed endpoint locations in the Camera_1 frame.
 * @param l1_sigma_list : The estimated standard deviation for projected endpoint locations.
 * @param stereo_cam : The stereo camera model.
 * @param T_10 : The pose of Camera_0 in Camera_1. Note that this is the inverse transform from
 *               Camera_0 to Camera_1. We will end up inverting this to estimate odometry.
 * @param H (output) : The Hessian at this linearization point.
 * @param g (output) : The gradient at this linearization point.
 * @param error (output) : The (weighted) sum of squared projection error.
 */
void LinearizeLineProjection(const std::vector<LineFeature3D> L0_list,
                             const std::vector<LineFeature2D> l1_obs_list,
                             const std::vector<double>& l1_sigma_list,
                             const StereoCamera& stereo_cam,
                             const Matrix4d& T_10,
                             Matrix6d& H,
                             Vector6d& g,
                             double& error);


int RemoveLineOutliers(const Matrix4d& T_10,
                       const std::vector<LineFeature3D>& L0_list,
                       const std::vector<LineFeature2D>& l1_obs_list,
                       const std::vector<double>& l1_sigma_list,
                       const StereoCamera& stereo_cam,
                       double max_err_stdevs,
                       std::vector<int>& inlier_indices);

}
}
