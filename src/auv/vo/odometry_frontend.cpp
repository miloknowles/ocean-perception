#include "vo/odometry_frontend.hpp"
#include "vo/feature_matching.hpp"
#include "vo/optimization.hpp"

#include "core/math_util.hpp"

#include "viz/visualize_matches.hpp"

namespace bm {
namespace vo {


OdometryFrontend::OdometryFrontend(const StereoCamera& stereo_camera,
                                   const Options& opt)
    : stereo_camera_(stereo_camera),
      camera_left_(stereo_camera.LeftIntrinsics()),
      camera_right_(stereo_camera.RightIntrinsics()),
      pdetector_(opt.point_detector),
      ldetector_(opt.line_detector)
{
}


OdometryEstimate OdometryFrontend::TrackStereoFrame(const Image1b& iml,
                                                    const Image1b& imr)
{

  //========================= STEREO MATCHING ============================
  std::vector<cv::KeyPoint> kpl, kpr;
  cv::Mat orbl, orbr;

  const int npl = pdetector_.Detect(iml, kpl, orbl);
  const int npr = pdetector_.Detect(imr, kpr, orbr);
  printf("[VO] Detected %d|%d keypoints in left|right images\n", npl, npr);

  std::vector<int> matches_lr;
  const int Np_stereo = StereoMatchPoints(
      kpl, orbl, kpr, orbr, stereo_camera_,
      opt_.stereo_max_epipolar_dist,
      opt_.stereo_min_distance_ratio,
      matches_lr);
  printf("[VO] Matched %d keypoints from left to right\n", Np_stereo);

  const auto& dmatches = viz::ConvertToDMatch(matches_lr);
  cv::Mat draw_stereo, draw_temporal;
  cv::drawMatches(iml, kpl, imr, kpr, dmatches, draw_stereo,
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imshow("stereo_matches", draw_stereo);

  const bool has_prev_frame = kpl_prev_.size() > 0;

  //======================= TEMPORAL MATCHING ==============================
  Matrix4d T_01 = Matrix4d::Identity();
  Matrix6d C_01 = Matrix6d::Identity();
  double error;
  std::vector<int> inlier_indices;

  if (has_prev_frame) {
    assert(kpl_prev_.size() == orbl_prev_.rows);

    std::vector<int> matches_01;
    const int Np_temporal = MatchPointsNN(orbl_prev_, orbl, opt_.temporal_min_distance_ratio, matches_01);

    assert(matches_01.size() == kpl_prev_.size());

    // 3D landmark points in the Camera_0 frame.
    std::vector<Vector3d> P0_list;
    std::vector<Vector2d> p1_list;
    std::vector<double> p1_sigma_list;

    for (int j0 = 0; j0 < matches_01.size(); ++j0) {
      const int j1 = matches_01.at(j0);
      if (j1 < 0) { continue; }

      const cv::KeyPoint kpl_j0 = kpl_prev_.at(j0);
      const cv::KeyPoint kpl_j1 = kpl.at(j1);

      const Vector2d pl0(kpl_j0.pt.x, kpl_j0.pt.y);
      const Vector2d pl1(kpl_j1.pt.x, kpl_j1.pt.y);

      const double disp = disp_prev_.at(j0);
      const double depth = camera_left_.fx() * stereo_camera_.Baseline() / std::max(1e-3, disp);

      P0_list.emplace_back(camera_left_.Backproject(pl0, depth));
      p1_list.emplace_back(pl1);
      p1_sigma_list.emplace_back(opt_.keypoint_sigma);
    }

    const int Ni = OptimizePoseIterativeP(
        P0_list, p1_list, p1_sigma_list, stereo_camera_, T_01, C_01, error, inlier_indices,
        opt_.opt_max_iters, opt_.opt_min_error,
        opt_.opt_min_error_delta, opt_.opt_max_error_stdevs);

    // Each item dmatches_01(i) links P0(i) and p1_obs(i) with its matching point.
    // Therefore, if p1_obs(i) is an inlier, we should keep dmatches_01.
    const std::vector<cv::DMatch> dmatches_01 = viz::ConvertToDMatch(matches_01);
    std::vector<char> matches_mask_01(dmatches_01.size());
    FillMask(inlier_indices, matches_mask_01);

    printf("[VO] Temporal matches: initial=%d | refined=%zu\n", Np_temporal, inlier_indices.size());
    cv::drawMatches(
        iml_prev_, kpl_prev_, iml, kpl, dmatches_01, draw_temporal,
        cv::Scalar::all(-1), cv::Scalar::all(-1), matches_mask_01,
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("temporal_matches", draw_temporal);
  }

  //====================== HOUSEKEEPING FOR PREVIOUS FRAME =========================
  // Store everything needed for temporal matching.
  kpl_prev_.resize(Np_stereo);
  kpr_prev_.resize(Np_stereo);
  orbl_prev_ = cv::Mat(Np_stereo, orbl.cols, orbl.type());
  orbr_prev_ = cv::Mat(Np_stereo, orbr.cols, orbr.type());
  disp_prev_.resize(Np_stereo);
  iml_prev_ = iml;

  // Only store the triangulated points.
  int match_ctr = 0;
  for (int j = 0; j < matches_lr.size(); ++j) {
    if (matches_lr.at(j) < 0) {
      continue;
    }

    const cv::KeyPoint kplj = kpl.at(j);
    const cv::KeyPoint kprj = kpr.at(matches_lr.at(j));

    disp_prev_.at(match_ctr) = std::fabs(kplj.pt.x - kprj.pt.x);
    kpl_prev_.at(match_ctr) = kplj;
    kpr_prev_.at(match_ctr) = kprj;

    // NOTE(milo): Important! Need to copy otherwise these rows will point to original data!
    orbl.row(j).copyTo(orbl_prev_.row(match_ctr));
    orbr.row(matches_lr.at(j)).copyTo(orbr_prev_.row(match_ctr));
    ++match_ctr;
  }

  //========================= RETURN ODOMETRY ESTIMATE ==========================
  OdometryEstimate out;
  out.T_l1_l0 = T_01.inverse();
  out.C_l1_l0 = C_01;
  out.error = error;
  out.tracked_keypoints = inlier_indices.size();
  out.tracked_keylines = 0;

  return out;
}

}
}
