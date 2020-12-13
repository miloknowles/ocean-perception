#pragma once

#include <string>

#include <opencv2/highgui/highgui.hpp>

#include "core/stereo_camera.hpp"
#include "core/pinhole_camera.hpp"
#include "core/file_utils.hpp"

namespace bm {
namespace dataset {


class StereoDataset {
 public:
  struct Options {
    std::string toplevel_path;
    std::string left_image_path  = "image_0";
    std::string right_image_path = "image_1";
  };

  StereoDataset(const Options& opt);

  int size() const { return left_filenames_.size(); }

  cv::Mat Left(int i, bool gray) const;
  cv::Mat Right(int i, bool gray) const;

 private:
  Options opt_;

  std::vector<std::string> left_filenames_;
  std::vector<std::string> right_filenames_;
};

}
}
