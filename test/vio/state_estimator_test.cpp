#include <gtest/gtest.h>
#include <glog/logging.h>
#include <utility>
#include <unordered_map>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>
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

typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;

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


// https://github.com/borglab/gtsam/pull/25
// https://github.com/borglab/gtsam/blob/d6b24294712db197096cd3ea75fbed3157aea096/gtsam_unstable/slam/tests/testSmartStereoFactor_iSAM2.cpp
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
  parameters.cacheLinearizedFactors = false; // TODO: added from example
  const double lag = 5.0;
  gtsam::IncrementalFixedLagSmoother fixed_lag_ISAM2(lag, parameters);

  gtsam::ISAM2 ISAM2(parameters);

   // Map: landmark_id => smart_factor_index inside iSAM2
  std::map<core::uid_t, gtsam::FactorIndex> lm2factor;

  std::unordered_map<core::uid_t, SmartStereoFactor::shared_ptr> stereo_factors_;

  const double skew = 0;
  gtsam::Cal3_S2Stereo::shared_ptr K_stereo_ptr_(
      new gtsam::Cal3_S2Stereo(
          stereo_rig.fx(),
          stereo_rig.fy(),
          skew,
          stereo_rig.cx(),
          stereo_rig.cy(),
          stereo_rig.Baseline())
  );

  const gtsam::noiseModel::Isotropic::shared_ptr stereo_factor_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, 3.0);

  //================================================================================================
  dataset::StereoCallback callback = [&](const StereoImage& stereo_data)
  {
    Matrix4d T_prev_cur_prior = Matrix4d::Identity();
    const StereoFrontend::Result& result = stereo_frontend.Track(stereo_data, T_prev_cur_prior);

    std::vector<core::uid_t> lmk_ids(result.lmk_obs.size());
    std::vector<Vector3d> t_world_lmks(result.lmk_obs.size());

    LOG(INFO) << "PROCESSING FRAME: " << result.camera_id << std::endl;

    // Special instructions for using iSAM2 + smart factors:
    // Must fill factorNewAffectedKeys:
    gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> factorNewAffectedKeys;

    // Map: factor index in the internal graph of iSAM2 => landmark_id
    std::map<gtsam::FactorIndex, core::uid_t> newFactor2lm;

    if (result.is_keyframe) {
      LOG(INFO) << "KEYFRAME" << std::endl;
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

        // Add stereo smart projection factors.
        if (!result.lmk_obs.empty()) {
          for (const LandmarkObservation& lmk_obs : result.lmk_obs) {
            const core::uid_t lmk_id = lmk_obs.landmark_id;

            // Creating factor for the first time.
            if (stereo_factors_.count(lmk_id) == 0) {
              stereo_factors_.emplace(lmk_id, new SmartStereoFactor(stereo_factor_noise_));

              // Indicate that the newest factor refers to lmk_id.
              // NOTE(milo): order matters here!
              newFactor2lm[new_factors.size()] = lmk_id;

              new_factors.push_back(stereo_factors_.at(lmk_id));
            } else {
              // Indicate that the ISAM2 factor now affects the camera pose with the current key.
              factorNewAffectedKeys[lm2factor.at(lmk_id)].insert(key);
            }

            SmartStereoFactor::shared_ptr stereo_ptr = stereo_factors_.at(lmk_id);
            const gtsam::StereoPoint2 stereo_point2(
                lmk_obs.pixel_location.x,                      // X-coord in left image
                lmk_obs.pixel_location.x - lmk_obs.disparity,  // x-coord in right image
                lmk_obs.pixel_location.y);                     // y-coord in both images (rectified)
            stereo_ptr->add(stereo_point2, key, K_stereo_ptr_);
          }
        }

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
        // fixed_lag_ISAM2.update(new_factors, new_values, new_timestamps);
        const gtsam::ISAM2Result isam_res1 = ISAM2.update(new_factors, new_values);

        // Figure out what factor index has been assigned to each new factor.
        for (const auto &f2l : newFactor2lm) {

          // TODO(milo): I think the problem is here ...
          lm2factor[f2l.second] = isam_res1.newFactorsIndices.at(f2l.first);
        }

        ISAM2.update();
        const gtsam::ISAM2Result isam_res2 = ISAM2.update();
        isam_res2.print();

        new_factors.resize(0);
        new_values.clear();
        new_timestamps.clear();

        // fixed_lag_ISAM2.update();
        // fixed_lag_ISAM2.update();

        // std::cout << "  iSAM2 Smoother Keys: " << std::endl;
        // for (const auto& key_timestamp: fixed_lag_ISAM2.timestamps()) {
        //   std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
        // }
        // std::cout << "  iSAM2 Smoother Keys: " << std::endl;
        // for (const auto& key_timestamp: ISAM2.timestamps()) {
        //   std::cout << std::setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << std::endl;
        // }
      }
    }
  };
  //================================================================================================

  dataset.RegisterStereoCallback(callback);
  dataset.Playback(10.0f, false);
  LOG(INFO) << "DONE" << std::endl;
}
