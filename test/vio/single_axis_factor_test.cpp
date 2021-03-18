#include <gtest/gtest.h>

#include "vio/gtsam_types.hpp"
#include "vio/single_axis_factor.hpp"

using namespace bm;
using namespace vio;


TEST(VioTest, TestSingleAxisFactor_01)
{
  const gtsam::Symbol pose1_sym('X', 0);
  const IsotropicModel::shared_ptr noise_model = IsotropicModel::Sigma(1, 3.0);

  const gtsam::SingleAxisFactor f(pose1_sym, core::Axis3::X, 123.456, noise_model);

  const gtsam::Pose3 pose1 = gtsam::Pose3::identity();

  gtsam::Matrix J;
  const gtsam::Vector1& error = f.evaluateError(pose1, J);

  std::cout << "J:\n" << J << std::endl;

  gtsam::Matrix16 expected_J;
  expected_J << 0, 0, 0, 1, 0, 0;

  EXPECT_EQ(expected_J, J);
  EXPECT_EQ(-123.456, error(0));
}
