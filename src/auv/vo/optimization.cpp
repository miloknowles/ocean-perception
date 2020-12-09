#include <eigen3/Eigen/QR>

#include "core/math_util.hpp"
#include "vo/optimization.hpp"

namespace bm {
namespace vo {


int OptimizePoseIterative(const std::vector<Vector3d>& P0_list,
                          const std::vector<Vector2d>& p1_obs_list,
                          const std::vector<double>& p1_sigma_list,
                          const StereoCamera& stereo_cam,
                          Matrix4d& T_01,
                          Matrix6d& C_01,
                          double& error,
                          std::vector<int>& inlier_indices,
                          int max_iters,
                          double min_error,
                          double min_error_delta,
                          double max_error_stdevs)
{
  // Do the initial pose optimization.
  const int N1 = OptimizePoseLevenbergMarquardt(
      P0_list, p1_obs_list, p1_sigma_list, stereo_cam,  // Inputs.
      T_01, C_01, error,                                // Outputs.
      max_iters, min_error, min_error_delta);           // Params.

  RemovePointOutliers(T_01, P0_list, p1_obs_list, p1_sigma_list,
                      stereo_cam, max_error_stdevs, inlier_indices);

  const std::vector<Vector3d>& P0_list_refined = Subset<Vector3d>(P0_list, inlier_indices);
  const std::vector<Vector2d>& p1_obs_list_refined = Subset<Vector2d>(p1_obs_list, inlier_indices);
  const std::vector<double>& p1_sigma_list_refined = Subset<double>(p1_sigma_list, inlier_indices);

  const int N2 = OptimizePoseLevenbergMarquardt(
      P0_list_refined, p1_obs_list_refined,
      p1_sigma_list_refined, stereo_cam,                // Inputs.
      T_01, C_01, error,                                // Outputs.
      max_iters, min_error, min_error_delta);           // Params.

  return N2;
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
                            double min_error_delta)
{
    // Set the initial guess (if not already set).
    if (T_01(3, 3) != 1.0) {
      T_01 = Matrix4d::Identity();
    }

    Matrix6d H;       // Current estimated Hessian of error w.r.t T_eps.
    Vector6d g;       // Current estimated gradient of error w.r.t T_eps.
    Vector6d T_eps;   // An incremental update to T_01.
    double err_prev = std::numeric_limits<double>::max();
    double err = err_prev - 1;

    int iters;
    for (iters = 0; iters < max_iters; ++iters) {
      LinearizePointProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_01, H, g, err);

      // Stop if error increases.
      if (err > err_prev) {
        break;
      }

      // Stop if error is small or hasn't changed much.
      if ((err < min_error) || (std::fabs(err - err_prev) < min_error_delta)) {
        break;
      }

      // Solve the equation H * T_eps = g.
      // T_eps will be the right-multiply transform that sets the error gradient to zero.
      Eigen::ColPivHouseholderQR<Matrix6d> solver(H);
      T_eps = solver.solve(g);

      // NOTE(milo): They seem to follow the '2nd' option on page 47.
      // See: https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
      T_01 << T_01 * inverse_se3(expmap_se3(T_eps));

      // Stop if the pose solution hasn't changed much.
      if (T_eps.head(3).norm() < min_error_delta && T_eps.tail(3).norm() < min_error_delta) {
        break;
      }

      err_prev = err;
    }

    C_01 = H.inverse();

    // Error of -1 indicates failure.
    error = iters > 0 ? err : -1;

    return iters;
}

int OptimizePoseLevenbergMarquardt(const std::vector<Vector3d>& P0_list,
                                  const std::vector<Vector2d>& p1_obs_list,
                                  const std::vector<double>& p1_sigma_list,
                                  const StereoCamera& stereo_cam,
                                  Matrix4d& T_01,
                                  Matrix6d& C_01,
                                  double& error,
                                  int max_iters,
                                  double min_error,
                                  double min_error_delta)
{
  // Set the initial guess (if not already set).
  if (T_01(3, 3) != 1.0) {
    T_01 = Matrix4d::Identity();
  }

  Matrix6d H;       // Current estimated Hessian of error w.r.t T_eps.
  Vector6d g;       // Current estimated gradient of error w.r.t T_eps.
  Vector6d T_eps;   // An incremental update to T_01.
  double err_prev = std::numeric_limits<double>::max();
  double err = err_prev - 1;

  const double lambda_k_increase = 2.0;
  const double lambda_k_decrease = 3.0;

  LinearizePointProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_01, H, g, err);

  double H_max = 0.0;
  for (int i = 0; i < 6; ++i) {
    if (H(i, i) > H_max || H(i, i) < -H_max) {
      H_max = std::fabs(H(i, i));
    }
  }
  double lambda  = 1e-5 * H_max;

  H.diagonal() += Vector6d::Constant(lambda);
  Eigen::ColPivHouseholderQR<Matrix6d> solver(H);
  T_eps = solver.solve(g);
  T_01 << T_01 * inverse_se3(expmap_se3(T_eps));
  err_prev = err;

  int iters;
  for (iters = 1; iters < max_iters; ++iters) {
    LinearizePointProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_01, H, g, err);

    if (err < min_error) {
      break;
    }

    H.diagonal() += Vector6d::Constant(lambda);

    Eigen::ColPivHouseholderQR<Matrix6d> solver(H);
    T_eps = solver.solve(g);

    // If error gets worse, want to increase the damping factor (more like gradient descent).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    if (err > err_prev) {
      lambda *= lambda_k_increase;

    // If error improves, decrease the damping factor (more like Gauss-Newton).
    // See: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    } else {
      lambda /= lambda_k_decrease;
      T_01 << T_01 * inverse_se3(expmap_se3(T_eps));
    }

    // Stop if the pose solution hasn't changed much.
    if (T_eps.head(3).norm() < min_error_delta && T_eps.tail(3).norm() < min_error_delta) {
      break;
    }

    err_prev = err;
  }

  error = err;
  C_01 = H.inverse();

  return iters;
}


void LinearizePointProjection(const std::vector<Vector3d>& P0_list,
                              const std::vector<Vector2d>& p1_obs_list,
                              const std::vector<double>& p1_sigma_list,
                              const StereoCamera& stereo_cam,
                              const Matrix4d& T_01,
                              Matrix6d& H,
                              Vector6d& g,
                              double& error)
{
  assert(P0_list.size() == p1_obs_list.size());
  assert(p1_obs_list.size() == p1_sigma_list.size());

  H = Matrix6d::Zero();    // Hessian for point projection error.
  g = Vector6d::Zero();    // Gradient for point projection error.
  error = 0.0;             // Line projection error.

  const PinholeCamera& cam = stereo_cam.LeftIntrinsics();

  // Add up projection errors from all associated points.
  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P1 = T_01.block<3, 3>(0, 0) * P0_list.at(i) + T_01.col(3).head(3);

    // Project P1 to a pixel location in Camera_1.
    const Vector2d p1 = cam.Project(P1);
    const Vector2d err = (p1 - p1_obs_list.at(i));
    const double err_norm = err.norm();

    // NOTE(milo): See page 54 for derivation of the Jacobian below.
    // https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    const double Px = P1(0);
    const double Py = P1(1);
    const double Pz = P1(2);
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
    const double p1_sigma = p1_sigma_list.at(i);
    const double residual = err_norm / p1_sigma;
    const double weight = RobustWeightCauchy(residual);

    // Update Hessian, Gradient, and Error.
    H += J * J.transpose() * weight;
    g += J * residual * weight;
    error += residual * residual * weight;
  }

  error /= static_cast<double>(P0_list.size());
}


void LinearizeLineProjection(const std::vector<LineFeature3D> L0_list,
                             const std::vector<LineFeature2D> l1_obs_list,
                             const std::vector<double>& l1_sigma_list,
                             const StereoCamera& stereo_cam,
                             const Matrix4d& T_01,
                             Matrix6d& H,
                             Vector6d& g,
                             double& error)
{
  assert(L0_list.size() == l1_obs_list.size());
  assert(l1_obs_list.size() == l1_sigma_list.size());

  H = Matrix6d::Zero();    // Hessian for line projection error.
  g = Vector6d::Zero();    // Gradient for line projection error.
  error = 0.0;             // Line projection error.

  const PinholeCamera& cam = stereo_cam.LeftIntrinsics();

  for (int i = 0; i < L0_list.size(); ++i) {
    const LineFeature3D& L0 = L0_list.at(i);
    const LineFeature2D& l1_obs = l1_obs_list.at(i);

    const Vector3d& Ps = T_01.block<3, 3>(0, 0) * L0.P_start + T_01.col(3).head(3);
    const Vector3d& Pe = T_01.block<3, 3>(0, 0) * L0.P_end   + T_01.col(3).head(3);

    const Vector2d ps = cam.Project(Ps);
    const Vector2d pe = cam.Project(Pe);

    const Vector3d& cross = l1_obs.cross;

    // NOTE(milo): I don't fully understand this part of the paper, but you can compute the distance
    // from a point to an infinite line using a cross product.
    // See: https://arxiv.org/pdf/1705.09479.pdf (Equation 6).
    Vector2d err;
    err << cross(0)*ps(0) + cross(1)*ps(1) + cross(2),
           cross(0)*pe(0) + cross(1)*pe(1) + cross(2);

    const double err_norm = err.norm();

    // Compute the Jacobian w.r.t the start point.
    double gx   = Ps(0);
    double gy   = Ps(1);
    double gz   = Ps(2);
    double gz2  = gz*gz;
    double fgz2 = cam.fx() / std::max(1e-7, gz2);
    double ds   = err(0);
    double de   = err(1);
    double lx   = cross(0);
    double ly   = cross(1);

    Vector6d Js;
    Js << + fgz2 * lx * gz,
          + fgz2 * ly * gz,
          - fgz2 * ( gx*lx + gy*ly ),
          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
          + fgz2 * ( gx*gz*ly - gy*gz*lx );

    // Compute the Jacobian w.r.t the end point.
    gx   = Pe(0);
    gy   = Pe(1);
    gz   = Pe(2);
    gz2  = gz*gz;
    fgz2 = cam.fx() / std::max(1e-7, gz2);
    Vector6d Je;
    Je << + fgz2 * lx * gz,
          + fgz2 * ly * gz,
          - fgz2 * ( gx*lx + gy*ly ),
          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
          + fgz2 * ( gx*gz*ly - gy*gz*lx );

    // Combine Jacobians from both points.
    const Vector6d J = (Js*ds + Je*de) / std::max(1e-7, err_norm);

    const double l1_sigma = l1_sigma_list.at(i);
    const double residual = err_norm / l1_sigma;
    double weight = RobustWeightCauchy(residual);

    // Weight the error of this line proportional to how "overlapped" the observation and
    // projection are. If the projected and observed line segments don't overlap at all, then this
    // error is probably not very informative (maybe outlier observation?) and is ignored.
    const double overlap_frac = LineSegmentOverlap(l1_obs.p_start, l1_obs.p_end, ps, pe);
    weight *= overlap_frac;

    // Update Hessian, Gradient, and Error.
    H += J * J.transpose() * weight;
    g += J * residual * weight;
    error += residual * residual * weight;
  }
}


int RemovePointOutliers(const Matrix4d& T_01,
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
  const PinholeCamera& cam = stereo_cam.LeftIntrinsics();

  for (int i = 0; i < P0_list.size(); ++i) {
    const Vector3d P1 = T_01.block<3, 3>(0, 0) * P0_list.at(i) + T_01.col(3).head(3);
    const Vector2d p1 = cam.Project(P1);
    const double e = (p1 - p1_obs_list.at(i)).norm();
    const double sigma = p1_sigma_list.at(i);
    if (e < (sigma * max_err_stdevs)) {
      inlier_indices.emplace_back(i);
    }
  }

  return inlier_indices.size();
}


int RemoveLineOutliers(const Matrix4d& T_01,
                       const std::vector<LineFeature3D>& L0_list,
                       const std::vector<LineFeature2D>& l1_obs_list,
                       const std::vector<double>& l1_sigma_list,
                       const StereoCamera& stereo_cam,
                       double max_err_stdevs,
                       std::vector<int>& inlier_indices)
{
  assert(L0_list.size() == l1_obs_list.size());
  assert(l1_obs_list.size() == l1_sigma_list.size());

  inlier_indices.clear();
  const PinholeCamera& cam = stereo_cam.LeftIntrinsics();

  for (int i = 0; i < L0_list.size(); ++i) {
    const LineFeature3D& L0 = L0_list.at(i);
    const LineFeature2D& l1_obs = l1_obs_list.at(i);

    const Vector3d& Ps = T_01.block<3, 3>(0, 0) * L0.P_start + T_01.col(3).head(3);
    const Vector3d& Pe = T_01.block<3, 3>(0, 0) * L0.P_end   + T_01.col(3).head(3);

    const Vector2d ps = cam.Project(Ps);
    const Vector2d pe = cam.Project(Pe);

    const Vector3d& cross = l1_obs.cross;

    // NOTE(milo): I don't fully understand this part of the paper, but you can compute the distance
    // from a point to an infinite line using a cross product.
    // See: https://arxiv.org/pdf/1705.09479.pdf (Equation 6).
    Vector2d err;
    err << cross(0)*ps(0) + cross(1)*ps(1) + cross(2),
           cross(0)*pe(0) + cross(1)*pe(1) + cross(2);

    const double e = err.norm();
    const double sigma = l1_sigma_list.at(i);

    // NOTE(milo): Using a simple endpoint error check for now. Looks like stvo-pl uses a metric
    // that's a little more complicated, but it was hard to understand. See if this works.
    if (e < (sigma * max_err_stdevs)) {
      inlier_indices.emplace_back(i);
    }
  }

  return inlier_indices.size();
}


}
}
