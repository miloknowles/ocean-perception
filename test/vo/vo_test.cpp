#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "core/file_utils.hpp"
#include "core/math_util.hpp"
#include "core/timer.hpp"
#include "dataset/stereo_dataset.hpp"
#include "viz/visualize_matches.hpp"
#include "vo/optimization.hpp"
#include "vo/point_detector.hpp"
#include "vo/feature_matching.hpp"
#include "vo/odometry_frontend.hpp"


using namespace bm::core;
using namespace bm::vo;
using namespace bm::viz;
using namespace bm::dataset;


TEST(VOTest, TestSeq01)
{
  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_camera(camera_model, camera_model, 0.2);

  OdometryFrontend::Options opt;
  opt.opt_max_error_stdevs = 3.0;
  OdometryFrontend frontend(stereo_camera, opt);

  StereoDataset::Options dopt;
  // dopt.toplevel_path = "/home/milo/datasets/Unity3D/farmsim/11_seafloor";
  dopt.toplevel_path = "/home/milo/datasets/Unity3D/farmsim/10_rockcircle";
  // dopt.toplevel_path = "/home/milo/datasets/Unity3D/farmsim/07_farm_lap";
  StereoDataset dataset(dopt);
  Matrix4d gt_T_w_0 = dataset.LeftPose(0);

  Matrix4d T_w_cam = gt_T_w_0;

  for (int ii = 0; ii < dataset.size(); ++ii) {
    printf("\n-----------------------------------FRAME #%d-------------------------------------\n", ii);
    const Image1b& iml = dataset.Left(ii, true);
    const Image1b& imr = dataset.Right(ii, true);

    const Image3b& rgbl = dataset.Left(ii, false);
    const Image3b& rgbr = dataset.Right(ii, false);

    double gt_sec;
    Quaterniond gt_q_w_cam;
    Vector3d gt_t_w_cam;
    dataset.LeftPose(ii, gt_sec, gt_q_w_cam, gt_t_w_cam);

    Timer timer(true);
    OdometryEstimate odom = frontend.TrackStereoFrame(iml, imr);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    if (odom.npoints_tracked < 3) {
      odom.T_0_1 = Matrix4d::Identity();
      std::cout << "[VO] LOST KEYPOINT TRACKING --> returning identity transform!" << std::endl;
    }

    T_w_cam = T_w_cam * odom.T_0_1;

    printf("Tracked keypoints = %d\n", odom.npoints_tracked);
    // std::cout << "Odometry estimate:\n" << odom.T_0_1 << std::endl;
    std::cout << "Avg. Reprojection Error: " << odom.error << std::endl;
    // std::cout << "Cov. estimate:\n" << odom.C_0_1 << std::endl;

    std::cout << "t_world_cam ESTIMATED:\n" << T_w_cam.block<3, 1>(0, 3) << std::endl;
    std::cout << "t_world_cam GROUNDTRUTH:\n" << gt_t_w_cam << std::endl;
    // std::cout << "q_world_cam ESTIMATED:\n" << Quaterniond(T_w_cam.block<3, 3>(0, 0)).coeffs() << std::endl;
    // std::cout << "q_world_cam GROUNDTRUTH:\n" << gt_q_w_cam.coeffs() << std::endl;

    const double t_err_m = (T_w_cam.block<3, 1>(0, 3) - gt_t_w_cam).norm();
    const double ang_err_deg = RadToDeg(gt_q_w_cam.angularDistance(Quaterniond(T_w_cam.block<3, 3>(0, 0))));

    printf("METRICS: trans_err=%.3lf (meters) || rot_err=%.3lf (deg)\n", t_err_m, ang_err_deg);

    cv::waitKey(0);
  }
}
