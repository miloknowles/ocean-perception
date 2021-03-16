#pragma once

#include <unordered_map>

#include "core/eigen_types.hpp"
#include "core/uid.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

namespace bm {
namespace vio {


// Preintegrated IMU types.
typedef gtsam::PreintegratedImuMeasurements Pim;
typedef gtsam::PreintegratedCombinedMeasurements PimC;
typedef gtsam::imuBias::ConstantBias ImuBias;

// Stereo/mono vision factors.
typedef gtsam::SmartStereoProjectionPoseFactor SmartStereoFactor;
typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartMonoFactor;

// Convenient map types.
typedef std::unordered_map<uid_t, SmartMonoFactor::shared_ptr> SmartMonoFactorMap;
typedef std::unordered_map<uid_t, SmartStereoFactor::shared_ptr> SmartStereoFactorMap;
typedef std::map<uid_t, gtsam::FactorIndex> LmkToFactorMap;

// Noise model types.
typedef gtsam::noiseModel::Isotropic IsotropicModel;
typedef gtsam::noiseModel::Diagonal DiagonalModel;

//================================= CONSTANTS ==================================
static const ImuBias kZeroImuBias = ImuBias(gtsam::Vector3::Zero(), gtsam::Vector3::Zero());
static const gtsam::Vector3 kZeroVelocity = gtsam::Vector3::Zero();
static const double kSetSkewToZero = 0.0;


}
}
