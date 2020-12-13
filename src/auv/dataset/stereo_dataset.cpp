#include "dataset/stereo_dataset.hpp"


namespace bm {
namespace dataset {

using namespace core;


StereoDataset::StereoDataset(const Options& opt) : opt_(opt)
{
  FilenamesInDirectory(Join(opt_.toplevel_path, opt_.left_image_path), left_filenames_, true);
  FilenamesInDirectory(Join(opt_.toplevel_path, opt_.right_image_path), right_filenames_, true);

  assert(left_filenames_.size() == right_filenames_.size());
}


cv::Mat StereoDataset::Left(int i, bool gray) const
{
  assert(i < size());
  return cv::imread(left_filenames_.at(i), gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}


cv::Mat StereoDataset::Right(int i, bool gray) const
{
  assert(i < size());
  return cv::imread(right_filenames_.at(i), gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}

}
}
