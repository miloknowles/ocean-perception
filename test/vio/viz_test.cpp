#include <gtest/gtest.h>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>


// https://answers.opencv.org/question/142473/how-can-i-register-a-keyboard-event-handier-for-opencv-viz/
void ExampleKeyboardCallback(const cv::viz::KeyboardEvent &w, void *t)
{
  cv::viz::Viz3d* viz_handle = (cv::viz::Viz3d*)t;
  if (w.action) {
    std::cout << "you pressed "<< w.code <<" = "<< w.symbol << " in viz window "<< viz_handle->getWindowName()<<"\n";
  }
}


TEST(VioTest, TestViz1)
{
  cv::viz::Viz3d viz("vio_window");

  // cv::viz::Color cloud_color_ = cv::viz::Color::white();
  // cv::viz::Color velocity_vector_color_ = cv::viz::Color::white();
  // cv::viz::Color velocity_prior_color_ = cv::viz::Color::red();
  // cv::viz::Color no_motion_prior_color_ = cv::viz::Color::cherry();
  // cv::viz::Color imu_to_left_cam_vector_color_ = cv::viz::Color::green();
  // cv::viz::Color left_cam_active_frustum_color_ = cv::viz::Color::green();
  // cv::viz::Color cam_with_linear_prior_frustum_color_ = cv::viz::Color::pink();
  // cv::viz::Color cam_with_pose_prior_frustum_color_ = cv::viz::Color::yellow();
  // cv::viz::Color btw_factor_color_ = cv::viz::Color::celestial_blue();

  viz.registerKeyboardCallback(ExampleKeyboardCallback, &viz);

  viz.setFullScreen(true);
  viz.setBackgroundColor(cv::viz::Color::black());

  cv::Mat1b im = cv::imread("./resources/farmsim_01_left.png", cv::IMREAD_GRAYSCALE);
  const cv::Matx33d K = {458.0, 0.0, im.cols / 2.0, 0.0, 458.0, im.rows / 2.0, 0.0, 0.0, 1.0};

  cv::viz::WCameraPosition f0(K, im, 1.0, cv::viz::Color::blue());
  cv::viz::WCameraPosition f1(K, im, 1.0, cv::viz::Color::green());
  cv::viz::WCameraPosition f2(K, im, 1.0, cv::viz::Color::red());

  cv::Affine3d pose0 = cv::Affine3d::Identity().translate(cv::Vec3d(0, 0, 0));
  cv::Affine3d pose1 = cv::Affine3d::Identity().translate(cv::Vec3d(0, 0, 1));
  cv::Affine3d pose2 = cv::Affine3d::Identity().translate(cv::Vec3d(0, 0, 2));

  viz.showWidget("f0", f0, pose0);
  viz.showWidget("f1", f1, pose1);
  viz.showWidget("f2", f2, pose2);

  viz.showWidget("origin", cv::viz::WCameraPosition());

  // std::vector<cv::Affine3d> poses = { pose0, pose1, pose2 };
  // cv::viz::WTrajectoryFrustums pose_sequence(poses, K, 1.0, cv::viz::Color::yellow());
  // viz.showWidget("poses", pose_sequence);


  std::vector<cv::Point3d> points = {
    cv::Point3d(1, 1, 1),
    cv::Point3d(2, 2, 2),
    cv::Point3d(3, 3, 3),
    cv::Point3d(3, 3, 5),
    cv::Point3d(3, 3, 7)
  };

  cv::viz::WPolyLine pline(points, cv::viz::Color::magenta());
  viz.showWidget("pline", pline);

  viz.setViewerPose(cv::Affine3d::Identity().translate(cv::Vec3d(0, 0, -1)));

  // NOTE(milo): Q, q, E, e to exit the window.
  viz.spin();

  LOG(INFO) << "Shutdown viz window" << std::endl;
}
