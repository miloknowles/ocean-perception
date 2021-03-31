#pragma once

#include "dataset/data_provider.hpp"

namespace bm {
namespace dataset {


class EurocDataset : public DataProvider {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Construct with a toplevel_path, which should contain the "mav0" folder inside of it.
  EurocDataset(const std::string& toplevel_path);

 private:
  void ParseStereo(const std::string& cam0_path, const std::string& cam1_path);

  void ParseImageFolder(const std::string& cam_folder,
                        std::vector<timestamp_t>& output_timestamps,
                        std::vector<std::string>& output_filenames);

  void ParseImu(const std::string& imu_csv_path);

  void ParseGroundtruth(const std::string& gt_path);

  void ParseDepth(const std::string& depth_csv_path);

  // Load in range measurements from a list of paths. This supports range measurements from
  // multiple receivers or beacons (e.g aps0 and aps1), which will all get put into the same
  // vector and sorted by timestamp.
  void ParseRange(const std::vector<std::string>& range_csv_paths);
};


}
}
