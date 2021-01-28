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
  // dopt.toplevel_path = "/home/milo/datasets/Unity3D/farmsim/03_forward_only";
  dopt.toplevel_path = "/home/milo/datasets/Unity3D/farmsim/01_rotate_lr";
  StereoDataset dataset(dopt);

  // const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/03_forward_only";
  // const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/06_seafloor_easy";
  // const std::string data_path = "/home/milo/datasets/Unity3D/farmsim/05_forward_side";
  Matrix4d gt_T_w_0 = dataset.LeftPose(0);

  Matrix4d T_0_cam = Matrix4d::Identity();

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
    printf("GROUNDTRUTH at t=%lf sec\n", gt_sec);
    std::cout << gt_t_w_cam << std::endl;
    std::cout << gt_q_w_cam.coeffs() << std::endl;

    Timer timer(true);
    OdometryEstimate odom = frontend.TrackStereoFrame(iml, imr);
    const double ms = timer.Elapsed().milliseconds();
    printf("Took %lf ms (%lf hz) to process frame\n", ms, 1000.0 / ms);

    if (odom.npoints_tracked < 3) {
      odom.T_0_1 = Matrix4d::Identity();
      std::cout << "[VO] Unreliable, setting identify transform" << std::endl;
    }

    T_0_cam = T_0_cam * odom.T_0_1;

    printf("Tracked keypoints = %d\n", odom.npoints_tracked);
    std::cout << "Odometry estimate:\n" << odom.T_0_1 << std::endl;
    std::cout << "Avg. reproj error:\n" << odom.error << std::endl;
    std::cout << "Pose estimate:\n" << gt_T_w_0 * T_0_cam << std::endl;
    std::cout << "Cov. estimate:\n" << odom.C_0_1 << std::endl;
    cv::waitKey(0);
  }
}
