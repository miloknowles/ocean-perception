#pragma once

#include "dataset/data_provider.hpp"

namespace bm {
namespace dataset {


class HimbDataset : public DataProvider {
 public:
  // Construct with a toplevel_path, which should contain the "train/test/val" folders inside.
  HimbDataset(const std::string& toplevel_path, const std::string& split_name);
};


}
}
