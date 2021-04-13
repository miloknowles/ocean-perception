#include <glog/logging.h>

#include "dataset/acfr_dataset.hpp"
#include "core/file_utils.hpp"

namespace bm {
namespace dataset {


AcfrDataset::AcfrDataset(const std::string& toplevel_path)
    : DataProvider()
{
  const std::string left_folder = Join(toplevel_path, "images/left");
  const std::string right_folder = Join(toplevel_path, "images/right");

  std::vector<std::string> left_imgs, right_imgs;
  FilenamesInDirectory(left_folder, left_imgs, true);
  FilenamesInDirectory(right_folder, right_imgs, true);

  CHECK_EQ(left_imgs.size(), left_imgs.size());

  for (size_t i = 0; i < left_imgs.size(); ++i) {
    const timestamp_t timestamp = 1e8 * i;
    stereo_data.emplace_back(timestamp, left_imgs.at(i), right_imgs.at(i));
  }

  SanityCheck();
}

}
}
