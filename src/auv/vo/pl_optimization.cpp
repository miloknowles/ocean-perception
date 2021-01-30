#include <eigen3/Eigen/QR>

#include "core/math_util.hpp"
#include "core/transform_util.hpp"
#include "vo/pl_optimization.hpp"
#include "vo/optimization.hpp"

namespace bm {
namespace vo {


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
                            double max_error_stdevs)
{
  // Do the initial pose optimization.
  const int N1 = OptimizeOdometryLML(
      P0_list, p1_obs_list, p1_sigma_list,              // Point inputs.
      L0_list, l1_obs_list, l1_sigma_list, stereo_cam,  // Line inputs.
      T_10, C_10, error,                                // Outputs.
      max_iters, min_error, min_error_delta);           // Params.

  RemovePointOutliers(T_10, P0_list, p1_obs_list, p1_sigma_list,
                      stereo_cam, max_error_stdevs, point_inlier_indices);

  RemoveLineOutliers(T_10, L0_list, l1_obs_list, l1_sigma_list,
                     stereo_cam, max_error_stdevs, line_inlier_indices);

  const std::vector<Vector3d>& P0_list_refined = Subset<Vector3d>(P0_list, point_inlier_indices);
  const std::vector<Vector2d>& p1_obs_list_refined = Subset<Vector2d>(p1_obs_list, point_inlier_indices);
  const std::vector<double>& p1_sigma_list_refined = Subset<double>(p1_sigma_list, point_inlier_indices);

  const std::vector<LineFeature3D>& L0_list_refined = Subset<LineFeature3D>(L0_list, line_inlier_indices);
  const std::vector<LineFeature2D>& l1_obs_list_refined = Subset<LineFeature2D>(l1_obs_list, line_inlier_indices);
  const std::vector<double>& l1_sigma_list_refined = Subset<double>(l1_sigma_list, line_inlier_indices);

  // Do a second pose optimization with only inlier features.
  const int N2 = OptimizeOdometryLML(
      P0_list_refined, p1_obs_list_refined, p1_sigma_list_refined,
      L0_list_refined, l1_obs_list_refined, l1_sigma_list_refined,
      stereo_cam, T_10, C_10, error,
      max_iters, min_error, min_error_delta);

  return N2;
}


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
                                    double min_error_delta)
{
  // Set the initial guess (if not already set).
  if (T_10(3, 3) != 1.0) {
    T_10 = Matrix4d::Identity();
  }

  Matrix6d H_p, H_l;       // Current estimated Hessian of error w.r.t T_eps.
  Vector6d g_p, g_l;       // Current estimated gradient of error w.r.t T_eps.
  Vector6d T_eps;          // An incremental update to T_10.
  double err_prev = std::numeric_limits<double>::max();
  double err_p, err_l;

  const double lambda_k_increase = 2.0;
  const double lambda_k_decrease = 3.0;

  LinearizeProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_10, H_p, g_p, err_p);
  LinearizeLineProjection(L0_list, l1_obs_list, l1_sigma_list, stereo_cam, T_10, H_l, g_l, err_l);

  Matrix6d H = H_p + H_l;
  Vector6d g = g_p + g_l;
  double err = err_p + err_l;

  const double H_max = MaxDiagonal(H);
  double lambda  = 1e-5 * H_max;

  H.diagonal() += Vector6d::Constant(lambda);
  Eigen::ColPivHouseholderQR<Matrix6d> solver(H);
  T_eps = solver.solve(g);
  T_10 << T_10 * inverse_se3(expmap_se3(T_eps));
  err_prev = err;

  int iters;
  for (iters = 1; iters < max_iters; ++iters) {
    LinearizeProjection(P0_list, p1_obs_list, p1_sigma_list, stereo_cam, T_10, H_p, g_p, err_p);
    LinearizeLineProjection(L0_list, l1_obs_list, l1_sigma_list, stereo_cam, T_10, H_l, g_l, err_l);

    H = H_p + H_l;
    g = g_p + g_l;
    err = err_p + err_l;

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
      T_10 << T_10 * inverse_se3(expmap_se3(T_eps));
    }

    // Stop if the pose solution hasn't changed much.
    if (T_eps.head(3).norm() < min_error_delta && T_eps.tail(3).norm() < min_error_delta) {
      break;
    }

    err_prev = err;
  }

  error = err;
  C_10 = H.inverse();

  return iters;
}


int RemoveLineOutliers(const Matrix4d& T_10,
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

    const Vector3d& Ps = T_10.block<3, 3>(0, 0) * L0.P_start + T_10.col(3).head(3);
    const Vector3d& Pe = T_10.block<3, 3>(0, 0) * L0.P_end   + T_10.col(3).head(3);

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


void LinearizeLineProjection(const std::vector<LineFeature3D> L0_list,
                             const std::vector<LineFeature2D> l1_obs_list,
                             const std::vector<double>& l1_sigma_list,
                             const StereoCamera& stereo_cam,
                             const Matrix4d& T_10,
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

    const Vector3d& Ps = T_10.block<3, 3>(0, 0) * L0.P_start + T_10.col(3).head(3);
    const Vector3d& Pe = T_10.block<3, 3>(0, 0) * L0.P_end   + T_10.col(3).head(3);

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

  error /= static_cast<double>(L0_list.size());
}

}
}
