#include "stereo_matching/stereo_matching.hpp"

namespace bm {
namespace stereo {


// static const int kNumDisp = 64;
// static const int kBlockSize = 3;


Image1f EstimateDisparity(const Image1b& il,
                          const Image1b& ir,
                          int num_disp,
                          int block_size)
{
  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, num_disp, block_size);

  const int channels = il.channels();

  // sgbm->setP1(8*channels*block_size*block_size);
  // sgbm->setP2(32*channels*block_size*block_size);
  sgbm->setMinDisparity(0);
  sgbm->setNumDisparities(num_disp);
  // sgbm->setUniquenessRatio(10);
  // sgbm->setSpeckleWindowSize(100);
  // sgbm->setSpeckleRange(32);
  // sgbm->setDisp12MaxDiff(1);

  sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

  cv::Mat disp;
  sgbm->compute(il, ir, disp);

  // https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html
  // StereoSGBM outputs a 16-bit fixed-point disparity map. This is the disparity value multiplied
  // by 16, so we need to divide by 16 to get back to disparity in pixels.
  Image1f dispf;
  disp.convertTo(dispf, CV_32F, 1.0f / 16.0f);

  return dispf;
}

}
}
