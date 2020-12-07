#pragma once

#include <vector>

#include "core/eigen_types.hpp"
#include "core/stereo_camera.hpp"

namespace bm {
namespace vo {

using namespace core;


// See: https://arxiv.org/pdf/1701.03077.pdf
inline double RobustWeightCauchy(double residual)
{
  return 1.0 / (1.0 + residual*residual);
}


int OptimizePoseGaussNewton(const std::vector<Vector3d>& P0_list,
                            const std::vector<Vector2d>& p1_obs_list,
                            const std::vector<double>& p1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_01,
                            Matrix6d& C_01,
                            double& error,
                            int max_iters,
                            double min_error,
                            double min_error_delta);

int OptimizePoseLevenbergMarquardt(const std::vector<Vector3d>& P0_list,
                            const std::vector<Vector2d>& p1_obs_list,
                            const std::vector<double>& p1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_01,
                            Matrix6d& C_01,
                            double& error,
                            int max_iters,
                            double min_error,
                            double min_error_delta);

void LinearizeProjection(const std::vector<Vector3d>& P0_list,
                         const std::vector<Vector2d>& p1_obs_list,
                         const std::vector<double>& p1_sigma_list,
                         const StereoCamera& stereo_cam,
                         const Matrix4d& T_01,
                         Matrix6d& H,
                         Vector6d& g,
                         double& error);

}
}
