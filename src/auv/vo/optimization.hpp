#pragma once

#include <vector>

#include "core/eigen_types.hpp"
#include "core/stereo_camera.hpp"
#include "core/line_feature.hpp"

namespace bm {
namespace vo {

using namespace core;


// See: https://arxiv.org/pdf/1701.03077.pdf
inline double RobustWeightCauchy(double residual)
{
  return 1.0 / (1.0 + residual*residual);
}


/**
 * Optimize the relative pose between two cameras using matched features. This pose is optimized
 * once, and then outlier features are removed before a refinement stage.
 *
 * @param[out] inlier_indices : The indices of inlier features in P0_list and p1_obs_list.
 */
int OptimizePoseIterativeP(const std::vector<Vector3d>& P0_list,
                          const std::vector<Vector2d>& p1_obs_list,
                          const std::vector<double>& p1_sigma_list,
                          const StereoCamera& stereo_cam,
                          Matrix4d& T_10,
                          Matrix6d& C_10,
                          double& error,
                          std::vector<int>& inlier_indices,
                          int max_iters,
                          double min_error,
                          double min_error_delta,
                          double max_error_stdevs);


int OptimizePoseIterativePL(const std::vector<Vector3d>& P0_list,
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


int OptimizePoseGaussNewtonP(const std::vector<Vector3d>& P0_list,
                            const std::vector<Vector2d>& p1_obs_list,
                            const std::vector<double>& p1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_10,
                            Matrix6d& C_10,
                            double& error,
                            int max_iters,
                            double min_error,
                            double min_error_delta);


int OptimizePoseLevenbergMarquardtP(const std::vector<Vector3d>& P0_list,
                            const std::vector<Vector2d>& p1_obs_list,
                            const std::vector<double>& p1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_10,
                            Matrix6d& C_10,
                            double& error,
                            int max_iters,
                            double min_error,
                            double min_error_delta);


int OptimizePoseLevenbergMarquardtPL(const std::vector<Vector3d>& P0_list,
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
 * Linearize image projection error around the SE3 manifold at the current pose.
 *
 * Camera_0 refers to the previous camera pose and Camera_1 refers to the current camera pose that
 * we are trying to estimate.
 *
 * @param P0_list : The 3D location of observed points (in the Camera_0) frame.
 * @param p1_obs_list : Observed keypoint locations in the Camera_1 image.
 * @param p1_sigma_list : Estimated standard deviation of projected keypoints locations (assume a
 *                        Gaussian noise model for projection, e.g 1px standard deviation).
 * @param stereo_cam : The stereo camera model for Camera_0 and Camera_1.
 * @param T_10 : The pose of Camera_0 in Camera_1. Note that this is the inverse transform from
 *               Camera_0 to Camera_1. We will end up inverting this to estimate odometry.
 * @param H (output) : The Hessian at this linearization point.
 * @param g (output) : The gradient at this linearization point.
 * @param error (output) : The (weighted) sum of squared projection error.
 */
void LinearizePointProjection(const std::vector<Vector3d>& P0_list,
                              const std::vector<Vector2d>& p1_obs_list,
                              const std::vector<double>& p1_sigma_list,
                              const StereoCamera& stereo_cam,
                              const Matrix4d& T_10,
                              Matrix6d& H,
                              Vector6d& g,
                              double& error);


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


int RemovePointOutliers(const Matrix4d& T_10,
                        const std::vector<Vector3d>& P0_list,
                        const std::vector<Vector2d>& p1_obs_list,
                        const std::vector<double>& p1_sigma_list,
                        const StereoCamera& stereo_cam,
                        double max_err_stdevs,
                        std::vector<int>& inlier_indices);


int RemoveLineOutliers(const Matrix4d& T_10,
                       const std::vector<LineFeature3D>& L0_list,
                       const std::vector<LineFeature2D>& l1_obs_list,
                       const std::vector<double>& l1_sigma_list,
                       const StereoCamera& stereo_cam,
                       double max_err_stdevs,
                       std::vector<int>& inlier_indices);

}
}
