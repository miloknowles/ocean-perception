#include "vo/odometry_frontend.hpp"
#include "vo/feature_matching.hpp"
#include "vo/optimization.hpp"

#include "core/math_util.hpp"
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
      pdetector_(opt.point_detector),
      ldetector_(opt.line_detector)
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

  //========================= STEREO LINE MATCHING =============================
  std::vector<ld::KeyLine> kll, klr;
  cv::Mat ldl, ldr;
  const int nl = ldetector_.Detect(iml, kll, ldl);
  const int nr = ldetector_.Detect(imr, klr, ldr);
  printf("[VO] Detected %d|%d LINES in left|right images\n", nl, nr);

  std::vector<int> lmatches_lr;
  const int Nl_stereo = StereoMatchLines(
      kll, klr, ldl, ldr, stereo_camera_,
      opt_.stereo_line_min_distance_ratio,
      std::cos(DegToRad(opt_.stereo_line_max_angle)),
      opt_.min_feature_disp,
      lmatches_lr);

  printf("Matched %d LINES from left to right\n", Nl_stereo);

  cv::Mat draw_stereo_lines, draw_temporal_lines;
  std::vector<cv::DMatch> lines_dmatches = viz::ConvertToDMatch(lmatches_lr);
  viz::DrawLineMatches(iml, kll, imr, klr, lines_dmatches, draw_stereo_lines,
                       std::vector<char>(), true);
  cv::imshow("stereo_lines", draw_stereo_lines);

  //======================= TEMPORAL MATCHING ==============================
  const bool has_prev_frame = kpl_prev_.size() > 0;

  Matrix4d T_01 = Matrix4d::Identity();
  Matrix6d C_01 = Matrix6d::Identity();
  double error;
  std::vector<int> point_inlier_indices, line_inlier_indices;

  if (has_prev_frame) {
    assert(kpl_prev_.size() == orbl_prev_.rows);
    assert(kll_prev_.size() == ldl_prev_.rows);

    // NOTE(milo): MatchPointsNN is way faster than TemporalMatchPoints, but has more false matches.
    std::vector<int> pmatches_01, lmatches_01;
    const int Np_temporal = MatchPointsNN(
        orbl_prev_, orbl, opt_.temporal_min_distance_ratio, pmatches_01);

    const int Nl_temporal = MatchLinesNN(
        ldl_prev_, ldl, NormalizedDirection(kll_prev_), NormalizedDirection(kll),
        opt_.temporal_line_min_distance_ratio,
        std::cos(DegToRad(opt_.stereo_line_max_angle)),
        lmatches_01);

    assert(pmatches_01.size() == kpl_prev_.size());
    assert(lmatches_01.size() == kll_prev_.size());

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

    //======================== 3D LINE LANDMARKS ===============================
    std::vector<LineFeature3D> L0_list;
    std::vector<LineFeature2D> l1_obs_list;
    std::vector<double> l1_sigma_list;

    for (int j0 = 0; j0 < lmatches_01.size(); ++j0) {
      const int j1 = lmatches_01.at(j0);
      if (j1 < 0) { continue; }

      L0_list.emplace_back(left_lines_prev_.at(j0));
      l1_obs_list.emplace_back(LineFeature2D(kll.at(j1)));
      l1_sigma_list.emplace_back(opt_.keyline_sigma);
    }

    if (opt_.track_lines) {
      const int Ni = OptimizePoseIterativePL(
          P0_list, p1_obs_list, p1_sigma_list,
          L0_list, l1_obs_list, l1_sigma_list,
          stereo_camera_, T_01, C_01, error,
          point_inlier_indices, line_inlier_indices,
          opt_.opt_max_iters, opt_.opt_min_error,
          opt_.opt_min_error_delta, opt_.opt_max_error_stdevs);
    } else {
      const int Ni = OptimizePoseIterativeP(
          P0_list, p1_obs_list, p1_sigma_list, stereo_camera_, T_01, C_01, error, point_inlier_indices,
          opt_.opt_max_iters, opt_.opt_min_error,
          opt_.opt_min_error_delta, opt_.opt_max_error_stdevs);
    }

    // Each item point_dm_01(i) links P0(i) and p1_obs(i) with its matching point.
    // Therefore, if p1_obs(i) is an inlier, we should keep point_dm_01.
    const std::vector<cv::DMatch> point_dm_01 = viz::ConvertToDMatch(pmatches_01);
    const std::vector<cv::DMatch> line_dm_01 = viz::ConvertToDMatch(lmatches_01);
    std::vector<char> point_mask_01(point_dm_01.size());
    std::vector<char> line_mask_01(line_dm_01.size());
    FillMask(point_inlier_indices, point_mask_01);
    FillMask(line_inlier_indices, line_mask_01);

    printf("[VO] Temporal POINT matches: initial=%d | refined=%zu\n", Np_temporal, point_inlier_indices.size());
    printf("[VO] Temporal LINE matches:  initial=%d | refined=%zu\n", Nl_temporal, line_inlier_indices.size());
    point_mask_01 = std::vector<char>(); // TODO
    cv::drawMatches(
        iml_prev_, kpl_prev_, iml, kpl, point_dm_01, draw_temporal_points,
        cv::Scalar::all(-1), cv::Scalar::all(-1), point_mask_01,
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    viz::DrawLineMatches(
        iml_prev_, kll_prev_, iml, kll, line_dm_01, draw_temporal_lines,
        line_mask_01, true);
    cv::imshow("temporal_points", draw_temporal_points);
    cv::imshow("temporal_lines", draw_temporal_lines);
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

  kll_prev_.resize(Nl_stereo);
  left_lines_prev_.resize(Nl_stereo);
  ldl_prev_ = cv::Mat(Nl_stereo, ldl.cols, ldl.type());

  // Only store the stereo-matched lines.
  match_ctr = 0;
  for (int j = 0; j < lmatches_lr.size(); ++j) {
    const int jr = lmatches_lr.at(j);

    if (jr < 0) { continue; }

    const ld::KeyLine& kllj = kll.at(j);
    const ld::KeyLine& klrj = klr.at(jr);

    LineSegment2d line_right_ext = ExtrapolateLineSegment(kllj, klrj);
    double disp_s, disp_e;
    ComputeEndpointDisparity(LineSegment2d(kllj), line_right_ext, disp_s, disp_e);

    const double depth_s = camera_left_.fx() * stereo_camera_.Baseline() / disp_s;
    const double depth_e = camera_left_.fx() * stereo_camera_.Baseline() / disp_e;
    const Vector3d& Ls = camera_left_.Backproject(Vector2d(kllj.startPointX, kllj.startPointY), depth_s);
    const Vector3d& Le = camera_left_.Backproject(Vector2d(kllj.endPointX, kllj.endPointY), depth_e);

    kll_prev_.at(match_ctr) = kll.at(j);
    left_lines_prev_.at(match_ctr) = LineFeature3D(Ls, Le);

    // NOTE(milo): Important! Need to copy otherwise these rows will point to original data!
    ldl.row(j).copyTo(ldl_prev_.row(match_ctr));

    ++match_ctr;
  }

  //========================= RETURN ODOMETRY ESTIMATE ==========================
  OdometryEstimate out;
  out.T_1_0 = inverse_se3(T_01);
  out.C_1_0 = C_01;
  out.error = error;
  out.tracked_keypoints = point_inlier_indices.size();
  out.tracked_keylines = line_inlier_indices.size();

  return out;
}

}
}
