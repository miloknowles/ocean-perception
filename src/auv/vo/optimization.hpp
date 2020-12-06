#pragma once

#include <vector>

#include <eigen3/Eigen/QR>

#include "core/math_util.hpp"
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
                            const std::vector<Vector2d>& xy1_obs_list,
                            const std::vector<double>& xy1_sigma_list,
                            const StereoCamera& stereo_cam,
                            Matrix4d& T_01,
                            Matrix6d& Sigma_01,
                            double& error,
                            int max_iters,
                            double min_error,
                            double min_error_delta)
{
    Matrix6d H;
    Vector6d g, dT;
    double err, err_prev = std::numeric_limits<double>::max();

    int iters;
    for (iters = 0; iters < max_iters; ++iters) {
        LinearizeProjection(P0_list, xy1_obs_list, xy1_sigma_list, stereo_cam, T_01, H, g, err);

        // Stop if error increases.
        if (err > err_prev) { break; }

        // Stop if error is small or hasn't changed much.
        if ((err < min_error) || (std::abs(err - err_prev) < min_error_delta)) {
          break;
        }

        // Solve the linearized system and update.
        Eigen::ColPivHouseholderQR<Matrix6d> solver(H);
        dT = solver.solve(g);

        // Update the pose.
        T_01 << T_01 * inverse_se3(expmap_se3(dT));
        // if the parameter change is small stop

        // Stop if the pose solution hasn't  changed much.
        if (dT.head(3).norm() < min_error_delta && dT.tail(3).norm() < min_error_delta) {
          break;
        }

        err_prev = err;
    }

    Sigma_01 = H.inverse();

    // Error of -1 indicates failure.
    error = iters > 0 ? err : -1;

    return iters;
}

void LinearizeProjection(const std::vector<Vector3d>& P0_list,
                         const std::vector<Vector2d>& xy1_obs_list,
                         const std::vector<double>& xy1_sigma_list,
                         const StereoCamera& stereo_cam,
                         const Matrix4d& T_01,
                         Matrix6d& H,
                         Vector6d& g,
                         double& error)
{
  assert(P0_list.size() == xy1_obs_list.size());

  Matrix6d H_p = Matrix6d::Zero();    // Hessian for point projection error.
  Matrix6d H_l = Matrix6d::Zero();    // Hessian for line projection error.
  Vector6d g_p = Vector6d::Zero();    // Gradient for point projection error.
  Vector6d g_l = Vector6d::Zero();    // Gradient for line projection error.
  double e_l = 0.0, e_p = 0.0;        // Line and point projection error.

  // Add up projection errors from all associated points.
  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P_curr = T_01 * P0_list.at(i);
    const Vector2d xy = stereo_cam.LeftIntrinsics().Project(P_curr);
    const Vector2d err = (xy - xy1_obs_list.at(i));
    const double err_norm = err.norm();

    const double Px = P_curr(0);
    const double Py = P_curr(1);
    const double Pz = P_curr(2);
    const double Pz_sq = Pz * Pz;
    const double fx = stereo_cam.fx();
    const double fx_Pz_sq = fx / std::max(1e-7, Pz_sq);   // Avoid divide by zero.
    const double ex = err(0);
    const double ey = err(1);

    Vector6d J;
    J << + fx_Pz_sq * ex * Pz,
         + fx_Pz_sq * ey * Pz,
         - fx_Pz_sq * (Px*ex + Py*ey),
         - fx_Pz_sq * (Px*Py*ex + Py*Py*ey + Pz*Pz*ey),
         + fx_Pz_sq * (Px*Px*ex + Pz*Pz*ex + Px*Py*ey),
         + fx_Pz_sq * (Px*Pz*ey - Py*Pz*ex);

    J = J / std::max(1e-7, err_norm);
    const double xy_sigma = xy1_sigma_list.at(i);
    const double residual = err_norm / xy_sigma;
    const double weight = RobustWeightCauchy(residual);

    // Update Hessian, Gradient, and Error.
    H_p += J * J.transpose() * weight;
    g_p += J * residual * weight;
    e_p += residual * residual * weight;
  }
}

}
}
