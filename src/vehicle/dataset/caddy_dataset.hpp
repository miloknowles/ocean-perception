#pragma once

#include "dataset/data_provider.hpp"

namespace bm {
namespace dataset {


class CaddyDataset : public DataProvider {
 public:
  // Construct with (2) paths:
  //  * "folder" points to the directory with splits in it
  //  * "split_name" is the name of the subfolder (e.g, "biograd_A")
  CaddyDataset(const std::string& folder, const std::string& split_name);
};


}
}
