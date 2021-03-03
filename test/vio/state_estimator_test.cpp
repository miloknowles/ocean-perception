#include <gtest/gtest.h>
#include <glog/logging.h>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>

#include "dataset/euroc_dataset.hpp"
#include "dataset/himb_dataset.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"
#include "core/uid.hpp"
#include "core/transform_util.hpp"
#include "vio/stereo_frontend.hpp"
#include "vio/visualization_2d.hpp"
#include "vio/visualizer_3d.hpp"
#include "vio/state_estimator.hpp"

using namespace bm;
using namespace core;
using namespace vio;


// TEST(VioTest, TestStateEstimator)
// {
//   const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
//   dataset::EurocDataset dataset(toplevel_folder);

//   const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
//   const StereoCamera stereo_rig(camera_model, 0.2);

//   StateEstimator::Options opt;
//   StateEstimator state_estimator(opt, stereo_rig);

//   // Matrix4d T_world_lkf = Matrix4d::Identity();

//   dataset.RegisterStereoCallback(std::bind(&StateEstimator::ReceiveStereo, &state_estimator, std::placeholders::_1));
//   dataset.Playback(10.0f, false);

//   // for (int i = 0; i < 30; ++i) {
//   //   dataset.Step();
//   // }

//   state_estimator.BlockUntilFinished();

//   LOG(INFO) << "DONE" << std::endl;
// }


TEST(VioTest, TestSimple1)
{
  const std::string toplevel_folder = "/home/milo/datasets/Unity3D/farmsim/euroc_test1";
  dataset::EurocDataset dataset(toplevel_folder);

  StereoFrontend::Options opt;

  const PinholeCamera camera_model(415.876509, 415.876509, 375.5, 239.5, 480, 752);
  const StereoCamera stereo_rig(camera_model, 0.2);
  StereoFrontend stereo_frontend(opt, stereo_rig);

  Matrix4d T_world_lkf = Matrix4d::Identity();
  gtsam::Key lkf_key(0);

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps;

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
  parameters.relinearizeSkip = 1; // Relinearize every time
  const double lag = 5.0;
  gtsam::IncrementalFixedLagSmoother smootherISAM2(lag, parameters);

  //================================================================================================
  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d T_prev_cur_prior = Matrix4d::Identity();
    const StereoFrontend::Result& result = stereo_frontend.Track(stereo_data, T_prev_cur_prior);

    std::vector<core::uid_t> lmk_ids(result.lmk_obs.size());
    std::vector<Vector3d> t_world_lmks(result.lmk_obs.size());

    LOG(INFO) << "PROCESSING FRAME: " << result.camera_id << std::endl;

    if (result.is_keyframe) {
      LOG(INFO) << "HIT KEYFRAME" << std::endl;
      T_world_lkf = T_world_lkf * result.T_lkf_cam;

      if (result.camera_id > 0) {
        const gtsam::Key key(result.camera_id);
        new_timestamps[key] = ConvertToSeconds(result.timestamp);

        // Add an initial guess for the camera pose based on raw visual odometry.
        const gtsam::Pose3 cam_pose(T_world_lkf);
        new_values.insert(key, cam_pose);

        // Add an odometry factor between prev and curr poses.
        const gtsam::Pose3 odom3(result.T_lkf_cam);
        const auto odom3_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.3)).finished());
        new_factors.push_back(gtsam::BetweenFactor<gtsam::Pose3>(lkf_key, key, odom3, odom3_noise));

        lkf_key = gtsam::Key(result.camera_id);

      } else {
        // Add prior on the first pose.
        gtsam::Pose3 prior_pose(Matrix4d::Identity());
        const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
                  (gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.3)).finished());
        const gtsam::Key key(result.camera_id);
        new_factors.addPrior(key, prior_pose, prior_noise);
        new_values.insert(key, gtsam::Pose3::identity());
        new_timestamps[key] = ConvertToSeconds(result.timestamp);
        lkf_key = key;
      }

      if (!new_values.empty()) {
        LOG(INFO) << "before update #1" << std::endl;
        smootherISAM2.getFactors().print();

        LOG(INFO) << "NEW FACTORS:" << std::endl;
        new_factors.print();
        const gtsam::IncrementalFixedLagSmoother::Result& isam_result = smootherISAM2.update(new_factors, new_values, new_timestamps);

        new_factors.resize(0);
        new_values.clear();
        new_timestamps.clear();

        // LOG(INFO) << "before update #2" << std::endl;
        // smootherISAM2.update();
        // LOG(INFO) << "before update #3" << std::endl;
        // smootherISAM2.update();

        std::cout << "  iSAM2 Smoother Keys: " << std::endl;
        for (const auto& key_timestamp: smootherISAM2.timestamps()) {
          std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
        }
      }
    }
  };
  //================================================================================================

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(10.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
