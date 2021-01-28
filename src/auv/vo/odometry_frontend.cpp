#include "vo/odometry_frontend.hpp"
#include "vo/feature_matching.hpp"
#include "vo/optimization.hpp"

#include "core/math_util.hpp"
#include "core/transform_util.hpp"
#include "core/line_segment.hpp"

#include "viz/visualize_matches.hpp"

namespace bm {
namespace vo {


OdometryFrontend::OdometryFrontend(const StereoCamera& stereo_camera,
                                   const Options& opt)
    : stereo_camera_(stereo_camera),
      camera_left_(stereo_camera.LeftIntrinsics()),
      camera_right_(stereo_camera.RightIntrinsics()),
      opt_(opt),
      pdetector_(opt.point_detector)
{
  std::cout << "[VO] Initialized OdometryFrontend" << std::endl;
}


OdometryEstimate OdometryFrontend::TrackStereoFrame(const Image1b& iml,
                                                    const Image1b& imr)
{
  //========================= STEREO POINT MATCHING ============================
  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat orbl, orbr;

  const int npl = pdetector_.Detect(iml, kpl, orbl);
  const int npr = pdetector_.Detect(imr, kpr, orbr);
  printf("[VO] Detected %d|%d POINTS in left|right images\n", npl, npr);

  std::vector<int> pmatches_lr;
  const int Np_stereo = StereoMatchPoints(
      kpl, orbl, kpr, orbr, stereo_camera_,
      opt_.stereo_max_epipolar_dist,
      opt_.stereo_min_distance_ratio,
      opt_.min_feature_disp,
      pmatches_lr);
  printf("[VO] Matched %d POINTS from left to right\n", Np_stereo);

  const auto& points_dmatches = viz::ConvertToDMatch(pmatches_lr);
  cv::Mat draw_stereo_points, draw_temporal_points;
  cv::drawMatches(iml, kpl, imr, kpr, points_dmatches, draw_stereo_points,
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imshow("stereo_points", draw_stereo_points);

  //======================= TEMPORAL MATCHING ==============================
  const bool has_prev_frame = kpl_prev_.size() > 0;

  Matrix4d T_10 = Matrix4d::Identity();
  Matrix6d C_10 = Matrix6d::Identity();
  double error;
  std::vector<int> point_inlier_indices, line_inlier_indices;

  if (has_prev_frame) {
    assert(kpl_prev_.size() == orbl_prev_.rows);

    // NOTE(milo): MatchPointsNN is way faster than TemporalMatchPoints, but has more false matches.
    std::vector<int> pmatches_01, lmatches_01;
    const int Np_temporal = MatchPointsNN(
        orbl_prev_, orbl, opt_.temporal_min_distance_ratio, pmatches_01);

    assert(pmatches_01.size() == kpl_prev_.size());

    //========================== 3D POINT LANDMARKS ============================
    std::vector<Vector3d> P0_list;
    std::vector<Vector2d> p1_obs_list;
    std::vector<double> p1_sigma_list;

    for (int j0 = 0; j0 < pmatches_01.size(); ++j0) {
      const int j1 = pmatches_01.at(j0);
      if (j1 < 0) { continue; }

      const cv::KeyPoint kpl_j0 = kpl_prev_.at(j0);
      const cv::KeyPoint kpl_j1 = kpl.at(j1);

      const Vector2d pl0(kpl_j0.pt.x, kpl_j0.pt.y);
      const Vector2d pl1(kpl_j1.pt.x, kpl_j1.pt.y);

      const double disp = disp_prev_.at(j0);
      const double depth = camera_left_.fx() * stereo_camera_.Baseline() / std::max(1e-3, disp);

      P0_list.emplace_back(camera_left_.Backproject(pl0, depth));
      p1_obs_list.emplace_back(pl1);
      p1_sigma_list.emplace_back(opt_.keypoint_sigma);
    }

    const int Ni = OptimizePoseIterativeP(
        P0_list, p1_obs_list, p1_sigma_list, stereo_camera_, T_10, C_10, error, point_inlier_indices,
        opt_.opt_max_iters, opt_.opt_min_error,
        opt_.opt_min_error_delta, opt_.opt_max_error_stdevs);

    // Each item point_dm_01(i) links P0(i) and p1_obs(i) with its matching point.
    // Therefore, if p1_obs(i) is an inlier, we should keep point_dm_01.
    const std::vector<cv::DMatch> point_dm_01 = viz::ConvertToDMatch(pmatches_01);
    std::vector<char> point_mask_01(point_dm_01.size());
    FillMask(point_inlier_indices, point_mask_01);

    printf("[VO] Temporal POINT matches: initial=%d | refined=%zu\n", Np_temporal, point_inlier_indices.size());
    cv::drawMatches(
        iml_prev_, kpl_prev_, iml, kpl, point_dm_01, draw_temporal_points,
        cv::Scalar::all(-1), cv::Scalar::all(-1), point_mask_01,
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("temporal_points", draw_temporal_points);
  }

  //====================== HOUSEKEEPING FOR PREVIOUS FRAME =========================
  // Store everything needed for temporal matching.
  kpl_prev_.resize(Np_stereo);
  orbl_prev_ = cv::Mat(Np_stereo, orbl.cols, orbl.type());
  disp_prev_.resize(Np_stereo);

  iml_prev_ = iml;

  int tmpctr = 0;
  for (const int jr : pmatches_lr) {
    if (jr >= 0) { ++tmpctr; }
  }
  // Only store the triangulated points.
  int match_ctr = 0;
  for (int j = 0; j < pmatches_lr.size(); ++j) {
    const int jr = pmatches_lr.at(j);

    if (jr < 0) { continue; }

    const cv::KeyPoint kplj = kpl.at(j);
    const cv::KeyPoint kprj = kpr.at(jr);

    disp_prev_.at(match_ctr) = std::fabs(kplj.pt.x - kprj.pt.x);
    kpl_prev_.at(match_ctr) = kplj;

    // NOTE(milo): Important! Need to copy otherwise these rows will point to original data!
    orbl.row(j).copyTo(orbl_prev_.row(match_ctr));
    ++match_ctr;
  }

  //========================= RETURN ODOMETRY ESTIMATE ==========================
  OdometryEstimate out;
  out.T_0_1 = inverse_se3(T_10);
  out.C_0_1 = C_10;
  out.error = error;
  out.tracked_keypoints = point_inlier_indices.size();

  return out;
}

}
}
