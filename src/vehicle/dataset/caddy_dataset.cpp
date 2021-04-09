#include <glog/logging.h>

#include "dataset/caddy_dataset.hpp"
#include "core/file_utils.hpp"

namespace bm {
namespace dataset {


CaddyDataset::CaddyDataset(const std::string& folder, const std::string& split_name)
    : DataProvider()
{
  const std::string split_folder = Join(folder, split_name);
  const std::string neg_folder = Join(split_folder, "true_negatives/raw");
  const std::string pos_folder = Join(split_folder, "true_positives/raw");

  std::vector<std::string> all_imgs;
  FilenamesInDirectory(neg_folder, all_imgs, false);
  FilenamesInDirectory(pos_folder, all_imgs, false);

  std::vector<std::string> left_imgs, right_imgs;

  // Sort out left and right images.
  for (const std::string& fname : all_imgs) {
    if (fname.find("left") != std::string::npos) {
      left_imgs.emplace_back(fname);
    } else if (fname.find("right") != std::string::npos) {
      right_imgs.emplace_back(fname);
    } else {
      LOG(FATAL) << "Image was neither left nor right: " << fname << std::endl;
    }
  }

  std::sort(left_imgs.begin(), left_imgs.end());
  std::sort(right_imgs.begin(), right_imgs.end());
  CHECK_EQ(left_imgs.size(), left_imgs.size());

  // Fake the images being taken at 10 Hz.
  for (size_t i = 0; i < left_imgs.size(); ++i) {
    const timestamp_t timestamp = 1e8 * i;
    stereo_data.emplace_back(timestamp, left_imgs.at(i), right_imgs.at(i));
  }

  SanityCheck();
}

}
}
