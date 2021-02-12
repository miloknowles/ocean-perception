#include <gtest/gtest.h>
#include <glog/logging.h>

#include "dataset/euroc_provider.hpp"

using namespace bm;
using namespace core;
using namespace dataset;


TEST(DatasetTest, TestEurocProvider)
{
  const std::string toplevel_folder = "/home/milo/datasets/euroc/V1_01_EASY";
  EurocProvider dataset(toplevel_folder);
}
