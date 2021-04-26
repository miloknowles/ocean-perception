#include "gtest/gtest.h"

#include "core/eigen_types.hpp"
#include "vision_core/pinhole_camera.hpp"
#include "vision_core/stereo_camera.hpp"

using namespace bm::core;


TEST(StereoCamera, TestConstruct)
{
  const PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  StereoCamera stereo_cam(cam, cam, 0.2);

  ASSERT_EQ(0.2, stereo_cam.Baseline());
  ASSERT_EQ(0.2, stereo_cam.Extrinsics().translation().x());

  const Transform3d T_right_left = stereo_cam.Extrinsics();
  stereo_cam =  StereoCamera(cam, cam, T_right_left);

  ASSERT_EQ(0.2, stereo_cam.Baseline());
  ASSERT_EQ(0.2, stereo_cam.Extrinsics().translation().x());
}

TEST(StereoCamera, TestProject)
{
  const PinholeCamera cam(415.876509, 415.876509, 376.0, 240.0, 480, 752);
  StereoCamera stereo_cam(cam, cam, 0.2);

  const Vector2d pl = stereo_cam.LeftCamera().Project(Vector3d(0, 0, 10));
  ASSERT_EQ(376.0, pl.x());
  ASSERT_EQ(240.0, pl.y());

  const Vector2d pl2 = stereo_cam.LeftCamera().Project(Vector3d(1, 2, 3));
  EXPECT_NEAR(376.0 + 415.876509 * 1 / 3, pl2.x(), 1e-3);
  EXPECT_NEAR(240.0 + 415.876509 * 2 / 3, pl2.y(), 1e-3);

  const Vector3d Pr = Vector3d(-3, 4, 10);
  const Vector2d pr = stereo_cam.RightCamera().Project(Vector3d(-3, 4, 10));
  ASSERT_EQ(Pr, stereo_cam.RightCamera().Backproject(pr, 10));
}
