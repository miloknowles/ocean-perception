#pragma once

#include "dataset/data_provider.hpp"

namespace bm {
namespace dataset {


class EurocProvider : public DataProvider {
 public:
  // Construct with a toplevel_path, which should contain the "mav0" folder inside of it.
  EurocProvider(const std::string& toplevel_path);

 private:
  void ParseStereo(const std::string& cam0_path, const std::string& cam1_path);

  void ParseImageFolder(const std::string& cam_folder,
                        std::vector<timestamp_t>& output_timestamps,
                        std::vector<std::string>& output_filenames);

  void ParseImu(const std::string& imu0_path);
};


}
}
