#pragma once

#include <string>

#include "core/imu_measurement.hpp"
#include "core/depth_measurement.hpp"
#include "core/stereo_image.hpp"

namespace bm {
namespace dataset {

using namespace core;

class EurocDataWriter {
 public:
  EurocDataWriter(const std::string& folder);

  void WriteImu(const ImuMeasurement& data);

  void WriteDepth(const DepthMeasurement& data);

  void WriteStereo(const StereoImage3b& data);

 private:
  std::string folder_;
  std::string imu_folder_;
  std::string depth_folder_;
  std::string left_folder_;
  std::string right_folder_;
};

}
}
