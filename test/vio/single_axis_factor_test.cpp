#include <gtest/gtest.h>
#include <glog/logging.h>

#include "core/eigen_types.hpp"
#include "vio/gtsam_types.hpp"
#include "vio/single_axis_factor.hpp"

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/slam/PartialPriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/base/numericalDerivative.h>

using namespace bm;
using namespace vio;
using namespace core;


TEST(VioTest, TestSingleAxisFactorJacobian) {
  gtsam::Key poseKey(1);
  gtsam::Pose3 measurement(gtsam::Rot3::RzRyRx(0.15, -0.30, 0.45), gtsam::Point3(-5.0, 8.0, -11.0));
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Isotropic::Sigma(1, 0.25);

  gtsam::SingleAxisFactor factor(poseKey, core::Axis3::Y, measurement.translation().y(), model);

  // Create a linearization point at the zero-error point
  gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.15, -0.30, 0.45), gtsam::Point3(-5.0, 8.0, -11.0));

  // Calculate numerical derivatives.
  gtsam::Matrix expectedH1 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
      boost::bind(&gtsam::SingleAxisFactor::evaluateError, &factor, _1, boost::none), pose);

  // Use the factor to calculate the derivative.
  gtsam::Matrix actualH1;
  factor.evaluateError(pose, actualH1);

  // Verify we get the expected error.
  CHECK(gtsam::assert_equal(expectedH1, actualH1, 1e-5));
}


TEST(VioTest, TestSingleAxisFactorGraph)
{
  gtsam::ISAM2 smoother;

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  const IsotropicModel::shared_ptr depth_noise = IsotropicModel::Sigma(1, 1.0);
  const IsotropicModel::shared_ptr pose_prior_noise = IsotropicModel::Sigma(6, 0.1);

  const gtsam::Symbol x0('X', 0);
  const gtsam::Symbol x1('X', 1);
  const gtsam::Symbol x2('X', 2);

  gtsam::Pose3 pose0((Matrix4d() << 1, 0, 0, 1.0,
                                    0, 1, 0, 35.0,
                                    0, 0, 1, -10.0,
                                    0, 0, 0, 1).finished());
  gtsam::Pose3 pose1((Matrix4d() << 1, 0, 0, 1.0,
                                    0, 0, 1, -20.0,
                                    0, -1, 0, -10.0,
                                    0, 0, 0, 1).finished());

  // POSE 0: Add prior and y-axis measurement.
  // The initial value has y=35.0, and here we add a measurement at y=40.0
  new_factors.push_back(gtsam::PriorFactor<gtsam::Pose3>(x0, pose0, pose_prior_noise));
  new_factors.push_back(gtsam::SingleAxisFactor(x0, core::Axis3::Y, 39.0, depth_noise));
  new_values.insert(x0, pose0);
  smoother.update(new_factors, new_values);

  const gtsam::Values& e0 = smoother.calculateBestEstimate();
  LOG(INFO) << "Pose0:\n";
  e0.at<gtsam::Pose3>(x0).print();

  new_factors.resize(0);
  new_values.clear();

  // POSE 1: Add prior and y-axis measurement.
  // The initial pose value has y=-20.0, and here we add a measurement at y=-25.0.
  new_factors.push_back(gtsam::SingleAxisFactor(x1, core::Axis3::Y, -12.0, depth_noise));
  new_values.insert(x1, pose1);
  smoother.update(new_factors, new_values);

  const gtsam::Values& e1 = smoother.calculateBestEstimate();
  LOG(INFO) << "Pose1:\n";
  e1.at<gtsam::Pose3>(x1).print();

  new_factors.resize(0);
  new_values.clear();
}


// https://gtsam.org/tutorials/intro.html#magicparlabel-65468
class UnaryFactor: public gtsam::NoiseModelFactor1<gtsam::Pose2> {
  double mx_, my_; ///< X and Y measurements

 public:
  UnaryFactor(gtsam::Key j, double x, double y, const gtsam::SharedNoiseModel& model):
    gtsam::NoiseModelFactor1<gtsam::Pose2>(model, j), mx_(x), my_(y) {}

  gtsam::Vector evaluateError(const gtsam::Pose2& q,
                              boost::optional<gtsam::Matrix&> H = boost::none) const
  {
    if (H) (*H) = (gtsam::Matrix(2,3)<< 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished();
    return (gtsam::Vector(2) << q.x() - mx_, q.y() - my_).finished();
  }
};


TEST(VioTest, TestTutorial_02)
{
  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;

  // NOTE(milo): Same as GTSAM tutorial, except all poses are rotated by 90 degrees.
  initial.insert(1, gtsam::Pose2(0.1, 10.0, M_PI_2));
  initial.insert(2, gtsam::Pose2(1.9, 10.0, M_PI_2));
  initial.insert(3, gtsam::Pose2(4.1, 10.0, M_PI_2));

  // Add a Gaussian prior on pose x_1.
  gtsam::Pose2 priorMean(0.0, 10.0, M_PI_2);
  gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
  graph.add(gtsam::PriorFactor<gtsam::Pose2>(1, priorMean, priorNoise));

  /**
   * The robot is moving in the +x direction, so odometry measurements will be in the -y direction.
   *
   *  WORLD FRAME    BODY FRAME (theta = 90 deg)
   *          (x)        (x)_______
   *           |                   |
   *           |                   |
   * (y) ______|                   |
   *                              (y)
   */
  gtsam::Pose2 odometry(0.0, -2.0, 0.0);
  gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.2, 0.2, 0.1));
  graph.add(gtsam::BetweenFactor<gtsam::Pose2>(1, 2, odometry, odometryNoise));
  graph.add(gtsam::BetweenFactor<gtsam::Pose2>(2, 3, odometry, odometryNoise));

  // add unary measurement factors, like GPS, on all three poses
  gtsam::noiseModel::Diagonal::shared_ptr unaryNoise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.1, 0.1)); // 10cm std on x,y
  graph.add(boost::make_shared<UnaryFactor>(1, 0.0, 9.0, unaryNoise));
  graph.add(boost::make_shared<UnaryFactor>(2, 2.0, 9.0, unaryNoise));
  graph.add(boost::make_shared<UnaryFactor>(3, 4.0, 9.0, unaryNoise));

  // optimize using Levenberg-Marquardt optimization
  gtsam::LevenbergMarquardtParams params;
  gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial, params).optimize();
  std::cout << "================================================================" << std::endl;
  std::cout << "result: " << std::endl;
  result.print();

  // Query the marginals
  std::cout.precision(2);
  gtsam::Marginals marginals(graph, result);
  std::cout << "================================================================" << std::endl;
  std::cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << std::endl;
  std::cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << std::endl;
  std::cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << std::endl;
}
