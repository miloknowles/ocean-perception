#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/types.hpp>

#include "vio/visualization_2d.hpp"

namespace bm {
namespace vio {


Image3b DrawFeatures(const Image1b& cur_img, const std::vector<cv::Point2f>& cur_keypoints)
{
  // Make some placeholder arguments so that we can use DrawFeatureTracks() on a single image.
  Image1b empty1;
  std::vector<cv::Point2f> empty2;
  return DrawFeatureTracks(empty1, cur_img, empty2, empty2, empty2, cur_keypoints);
}


Image3b DrawFeatureTracks(const Image1b& ref_img,
                          const Image1b& cur_img,
                          const std::vector<cv::Point2f>& ref_keypoints,
                          const std::vector<cv::Point2f>& cur_keypoints,
                          const std::vector<cv::Point2f>& untracked_ref,
                          const std::vector<cv::Point2f>& untracked_cur)
{
  Image3b img_bgr(cur_img.size(), CV_8U);

  cv::cvtColor(cur_img, img_bgr, cv::COLOR_GRAY2BGR);

  // Optionally show corners that were only detected in ref/cur.
  for (const auto& kp : untracked_ref) {
    cv::circle(img_bgr, kp, 4, cv::viz::Color::red(), 1);
  }
  for (const auto& kp : untracked_cur) {
    cv::circle(img_bgr, kp, 4, cv::viz::Color::blue(), 1);
  }

  // Show tracks from ref_img to cur_img.
  for (size_t i = 0; i < ref_keypoints.size(); ++i) {
    const cv::Point& px_cur = cur_keypoints.at(i);
    const cv::Point& px_ref = ref_keypoints.at(i);

    cv::circle(img_bgr, px_cur, 6, cv::viz::Color::green(), 1);
    cv::arrowedLine(img_bgr, px_ref, px_cur, cv::viz::Color::green(), 1);
  }

  return img_bgr;
}


}
}
