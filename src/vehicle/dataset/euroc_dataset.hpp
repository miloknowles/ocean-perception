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

  void ParseRange(const std::string& range_csv_path);
};


}
}
