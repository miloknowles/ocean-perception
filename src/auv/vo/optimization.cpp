#include <eigen3/Eigen/QR>

#include "core/math_util.hpp"
#include "core/transform_util.hpp"
#include "vo/optimization.hpp"

namespace bm {
namespace vo {


int OptimizeOdometryIterative(const std::vector<Vector3d>& P0_list,
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
                              double max_error_stdevs)
{
  // Do the initial pose optimization.
  const int N1 = OptimizeOdometryLM(
      P0_list, p1_obs_list, p1_sigma_list, stereo_cam,  // Inputs.
      T_10, C_10, error,                                // Outputs.
      max_iters, min_error, min_error_delta);           // Params.

  RemovePointOutliers(T_10, P0_list, p1_obs_list, p1_sigma_list,
                      stereo_cam, max_error_stdevs, inlier_indices);

  if (inlier_indices.size() < 6) {
    T_10 = Matrix4d::Identity();
    C_10 = Matrix6d::Identity();
    return -1;
  }

  const std::vector<Vector3d>& P0_list_refined = Subset<Vector3d>(P0_list, inlier_indices);
  const std::vector<Vector2d>& p1_obs_list_refined = Subset<Vector2d>(p1_obs_list, inlier_indices);
  const std::vector<double>& p1_sigma_list_refined = Subset<double>(p1_sigma_list, inlier_indices);

  const int N2 = OptimizeOdometryLM(
      P0_list_refined, p1_obs_list_refined,
      p1_sigma_list_refined, stereo_cam,                // Inputs.
      T_10, C_10, error,                                // Outputs.
      max_iters, min_error, min_error_delta);           // Params.

  return N2;
}


static double ComputeProjectionError(const std::vector<Vector3d>& P0_list,
                                    const std::vector<Vector2d>& p1_obs_list,
                                    const std::vector<double>& p1_sigma_list,
                                    const StereoCamera& stereo_cam,
                                    const Matrix4d& T_10)
{
  assert(P0_list.size() == p1_obs_list.size());
  assert(p1_obs_list.size() == p1_sigma_list.size());

  const int M = P0_list.size();

  double error = 0.0;

  const PinholeCamera& cam = stereo_cam.LeftCamera();

  // Add up projection errors from all associated points.
  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P1 = T_10.block<3, 3>(0, 0)*P0_list.at(i) + T_10.col(3).head(3);

    // Project P1 to a pixel location in Camera_1.
    const Vector2d p = cam.Project(P1);
    const Vector2d p_hat = p1_obs_list.at(i);
    const double rx = p_hat.x() - p.x();
    const double ry = p_hat.y() - p.y();
    const double r2 = rx*rx + ry*ry;
    const double r = std::sqrt(r2);
    const double sigma = p1_sigma_list.at(i);
    const double r_sigma = r / sigma;

    error += r_sigma;
  }

  return error / static_cast<double>(M);
}


int OptimizeOdometryLM(const std::vector<Vector3d>& P0_list,
                                  const std::vector<Vector2d>& p1_obs_list,
                                  const std::vector<double>& p1_sigma_list,
                                  const StereoCamera& stereo_cam,
                                  Matrix4d& T_10,
                                  Matrix6d& C_10,
                                  double& error,
                                  int max_iters,
                                  double min_error,
                                  double min_error_delta)
{
  // Set the initial guess (if not already set).
  if (T_10(3, 3) != 1.0) {
    T_10 = Matrix4d::Identity();
  }

  Matrix6d H;       // Current estimated Hessian of error w.r.t T_eps.
  Vector6d g;       // Current estimated gradient of error w.r.t T_eps.
  Vector6d T_eps;   // An incremental update to T_10.
  double err_prev = 123;
  double err = 123;

  const double lambda_k_increase = 2.0;
  const double lambda_k_decrease = 3.0;

  LinearizeProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_10, H, g, err);
  err_prev = err + 1;

  // https://arxiv.org/pdf/1201.5885.pdf
  double lambda = 8e-2;

  int iters;
  for (iters = 0; iters < max_iters; ++iters) {
    Matrix6d H_lm = H;
    H_lm.diagonal() += lambda*H.diagonal();
    Eigen::ColPivHouseholderQR<Matrix6d> solver(H_lm);
    T_eps = solver.solve(g);

    // Check if applying T_eps would improve error.
    const Matrix4d T_10_test = expmap_se3(T_eps) * T_10;
    err = ComputeProjectionError(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_10_test);

    if (err < min_error) {
      break;
    }

    // If error gets worse, want to increase the damping factor (more like gradient descent).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    if (err >= err_prev) {
      lambda *= lambda_k_increase;
      // printf("Error increased, increasing lambda (prev=%lf, err=%lf, lambda=%lf)\n", err_prev, err, lambda);

    // If error improves, decrease the damping factor (more like Gauss-Newton).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    } else {
      lambda /= lambda_k_decrease;
      // printf("Error decreased, decreasing lambda (prev=%lf, err=%lf, lambda=%lf)\n", err_prev, err, lambda);
      err_prev = err;

      T_10 = T_10_test;

      // Need to re-linearize because we updated T_10.
      LinearizeProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_10, H, g, err);
    }
  }

  error = err;
  C_10 = H.inverse();

  return iters;
}


void LinearizeProjection(const std::vector<Vector3d>& P0_list,
                          const std::vector<Vector2d>& p1_obs_list,
                          const std::vector<double>& p1_sigma_list,
                          const StereoCamera& stereo_cam,
                          const Matrix4d& T_10,
                          Matrix6d& H,
                          Vector6d& g,
                          double& error)
{
  assert(P0_list.size() == p1_obs_list.size());
  assert(p1_obs_list.size() == p1_sigma_list.size());

  const int M = P0_list.size();

  // Initialize Jacobian w/ zeros.
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M, 6);

  // Initialize residual error vector R.
  Eigen::VectorXd R = Eigen::VectorXd::Zero(M);

  error = 0.0;             // Line projection error.

  const PinholeCamera& cam = stereo_cam.LeftCamera();

  // Add up projection errors from all associated points.
  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P1 = T_10.block<3, 3>(0, 0)*P0_list.at(i) + T_10.col(3).head(3);

    // TODO: filter out points that project behind the camera...

    // Project P1 to a pixel location in Camera_1.
    const Vector2d p = cam.Project(P1);
    const Vector2d p_hat = p1_obs_list.at(i);
    const double rx = p_hat.x() - p.x();
    const double ry = p_hat.y() - p.y();
    const double r2 = rx*rx + ry*ry;
    const double r = std::sqrt(r2);
    const double sigma = p1_sigma_list.at(i);
    const double r_sigma = r / sigma;
    const double weight = RobustWeightCauchy(r_sigma);

    const double chain_rule_terms = -weight / std::max(1e-5, sigma*r);

    // NOTE(milo): See page 54 for derivation of the Jacobian below.
    // https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    const double gx = P1.x();
    const double gy = P1.y();
    const double gz = std::max(1e-5, P1.z());
    const double gz2 = gz*gz;
    const double fx = stereo_cam.fx();
    const double fy = stereo_cam.fy();

    Vector6d Ji;
    Ji << + rx*fx / gz,
          + ry*fy / gz,
          - (rx*fx*gx + ry*fy*gy) / gz2,
          - rx*fx*gx*gy/gz2 - ry*fy*(1.0 + gy*gy/gz2),
          + rx*fx*(1.0 + gx*gx/gz2) + ry*fy*gx*gy/gz2,
          - rx*fx*gy/gz + ry*fy*gx/gz;

    J.row(i) = chain_rule_terms * Ji;
    R(i) = weight * r_sigma;
    error += r_sigma;
  }

  H = J.transpose() * J;
  g = -J.transpose() * R;

  // Compute the AVERAGE error across all points.
  error /= static_cast<double>(M);
}

int RemovePointOutliers(const Matrix4d& T_10,
                        const std::vector<Vector3d>& P0_list,
                        const std::vector<Vector2d>& p1_obs_list,
                        const std::vector<double>& p1_sigma_list,
                        const StereoCamera& stereo_cam,
                        double max_err_stdevs,
                        std::vector<int>& inlier_indices)
{
  assert(P0_list.size() == p1_obs_list.size());
  assert(p1_obs_list.size() == p1_sigma_list.size());

  inlier_indices.clear();
  const PinholeCamera& cam = stereo_cam.LeftCamera();

  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P1 = T_10.block<3, 3>(0, 0) * P0_list.at(i) + T_10.col(3).head(3);
    const Vector2d p1 = cam.Project(P1);
    const double e = (p1 - p1_obs_list.at(i)).norm();
    const double sigma = p1_sigma_list.at(i);
    if (e < (sigma * max_err_stdevs)) {
      inlier_indices.emplace_back(i);
    }
  }

  return inlier_indices.size();
}


}
}
