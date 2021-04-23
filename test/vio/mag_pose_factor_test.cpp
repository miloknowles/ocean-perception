#include <gtest/gtest.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

// #include <CppUnitLite/TestHarness.h>

#include "core/mag_measurement.hpp"
#include "vio/mag_pose_factor.hpp"

using namespace bm;
using namespace core;
using namespace gtsam;

// Magnetic field in the nav frame (NED).
Point3 nM(22653.29982, -1956.83010, 44202.47862);

// Assumed scale factor (scales unit vector to magnetometer units of nT).
double scale = 255.0 / 50000.0;

// Ground truth Pose2/Pose3 in the nav frame.
Pose3 n_P3_b = Pose3(Rot3::Yaw(-0.1), Point3(-3, 12, 5));
Pose2 n_P2_b = Pose2(Rot2(-0.1), Point2(-3, 12));
Rot3 nRb = n_P3_b.rotation();
Rot2 theta = n_P2_b.rotation();

// Sensor bias (nT).
Point3 bias3(10, -10, 50);
Point2 bias2 = bias3.head(2);

// double s(scale * nM.norm());
Point2 dir2(nM.head(2).normalized());
Point3 dir3(nM.normalized());

// Compute the measured field in the sensor frame.
Point3 measured3 = nRb.inverse() * (scale * dir3) + bias3;

// The 2D magnetometer will measure the "NE" field components.
Point2 measured2 = theta.inverse() * (scale * dir2) + bias2;

SharedNoiseModel model2 = noiseModel::Isotropic::Sigma(2, 0.25);
SharedNoiseModel model3 = noiseModel::Isotropic::Sigma(3, 0.25);


//******************************************************************************
TEST(MagPoseFactor, Constructors)
{
  MagPoseFactor<Pose2> f2(Symbol('X', 0), measured2, scale, dir2, bias2, model2, boost::none);
  MagPoseFactor<Pose3> f3(Symbol('X', 0), measured3, scale, dir3, bias3, model3, boost::none);
}

//******************************************************************************
TEST(MagPoseFactor, Jacobians)
{
  Matrix H2, H3;

  // Error should be zero at the groundtruth pose.
  MagPoseFactor<Pose2> f2(Symbol('X', 0), measured2, scale, dir2, bias2, model2, boost::none);
  EXPECT_TRUE(gtsam::assert_equal(Z_2x1, f2.evaluateError(n_P2_b, H2), 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(gtsam::numericalDerivative11<Vector, Pose2> //
      (boost::bind(&MagPoseFactor<Pose2>::evaluateError, &f2, _1, boost::none), n_P2_b), H2, 1e-7));

  // Error should be zero at the groundtruth pose.
  MagPoseFactor<Pose3> f3(Symbol('X', 0), measured3, scale, dir3, bias3, model3, boost::none);
  EXPECT_TRUE(gtsam::assert_equal(Z_3x1, f3.evaluateError(n_P3_b, H3), 1e-5));
  EXPECT_TRUE(gtsam::assert_equal(gtsam::numericalDerivative11<Vector, Pose3> //
      (boost::bind(&MagPoseFactor<Pose3>::evaluateError, &f3, _1, boost::none), n_P3_b), H3, 1e-7));
}

// *************************************************************************
// int main() {
//   TestResult tr;
//   return TestRegistry::runAllTests(tr);
// }
// *************************************************************************
