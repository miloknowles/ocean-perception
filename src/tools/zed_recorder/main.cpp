#include <glog/logging.h>

#include "zed_recorder.hpp"

using namespace bm;
using namespace zed;


int main(int argc, char const *argv[])
{
  // Set up glog.
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  const char* path = std::getenv("BM_DATASETS_DIR");
  CHECK(path != nullptr) << "No environment variable $BM_DATASETS_DIR. Did you source setup.bash?" << std::endl;

  const std::string datasets_path(path);
  ZedRecorder zr(datasets_path);
  zr.Run(true);
  return 0;
}
