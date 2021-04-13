#pragma once

#include "dataset/data_provider.hpp"

namespace bm {
namespace dataset {


class AcfrDataset : public DataProvider {
 public:
  AcfrDataset(const std::string& toplevel_path);
};


}
}
