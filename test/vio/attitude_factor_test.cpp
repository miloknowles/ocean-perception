#include <gtest/gtest.h>
#include <glog/logging.h>

#include "core/eigen_types.hpp"
#include "vio/noise_model.hpp"

#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/slam/PartialPriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/AttitudeFactor.h>

using namespace bm;
using namespace vio;
using namespace core;


TEST(VioTest, TestAttitudeFactorGraph)
{
  gtsam::ISAM2 smoother;

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;

  const IsoModel::shared_ptr pose_prior_noise = IsoModel::Sigma(6, 0.1);

  const gtsam::Symbol pose_sym('X', 0);

  // Rotation corresponds to a +45 deg rotation around +z.
  gtsam::Pose3 world_P_body((Matrix4d() << 0.7071068, -0.7071068,  0.0000000,   17.1,
                                           0.7071068,  0.7071068,  0.0000000,   2,
                                           0.0000000,  0.0000000,  1.0000000,  -3,
                                           0,          0,          0,           1).finished());

  // POSE 0: Add prior and y-axis measurement.
  // The initial value has y=35.0, and here we add a measurement at y=40.0
  new_factors.push_back(gtsam::PriorFactor<gtsam::Pose3>(pose_sym, world_P_body, pose_prior_noise));

  // Using an RDF body frame.
  // Add an attitude factor: observe a gravity vector that points 45 deg between x and y axes.
  // This means that the body has rotate +45 deg around the +z axis (roll to the right).
  const Vector3d body_nG_unit = Vector3d(1, 1, 0).normalized();

  // Gravity points in the +y direction in the world frame (RDF).
  const Vector3d world_nG_unit = Vector3d(0, 9.81, 0).normalized();

  const IsoModel::shared_ptr attitude_noise = IsoModel::Sigma(2, 0.1);

  // NOTE(milo): GTSAM computes error as: nZ_.error(nRb * bRef_).
  // So if we measure body_nG, we should plug it in for bRef_, and use the world_nG as nZ_.
  const gtsam::Pose3AttitudeFactor attitude_factor(
      pose_sym,
      gtsam::Unit3(world_nG_unit),
      attitude_noise,
      gtsam::Unit3(body_nG_unit));
  new_factors.push_back(attitude_factor);

  // Make the initial guess slightly wrong (40 deg rotation instead of 45).
  gtsam::Pose3 world_P_body_guess(gtsam::Rot3(
    (Matrix3d() << 0.7660444, -0.6427876,  0.0000000,
                   0.6427876,  0.7660444,  0.0000000,
                   0.0000000,  0.0000000,  1.0000000).finished()),
    world_P_body.translation());
  new_values.insert(pose_sym, world_P_body_guess);

  smoother.update(new_factors, new_values);

  const gtsam::Values& estimate = smoother.calculateBestEstimate();
  LOG(INFO) << "Pose:\n";
  estimate.at<gtsam::Pose3>(pose_sym).print();

  // Check that the original pose is recovered.
  gtsam::assert_equal(world_P_body, estimate.at<gtsam::Pose3>(pose_sym), 1e-3);
}
