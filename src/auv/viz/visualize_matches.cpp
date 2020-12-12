#include <unordered_set>

#include "visualize_matches.hpp"

namespace bm {
namespace viz {


std::vector<cv::DMatch> ConvertToDMatch(const std::vector<int>& matches12)
{
  std::vector<cv::DMatch> out;
  for (int i = 0; i < matches12.size(); ++i) {
    if (matches12.at(i) < 0) { continue; }

    // NOTE(milo): queryIdx, trainIdx, distance. 'query' refers to an item in 1, and 'train' refers
    // to an item in 2. I'm not exactly sure why they're named that way.
    out.emplace_back(i, matches12.at(i), 0);
  }

  return out;
}

void FillMask(const std::vector<cv::DMatch>& matches_12, const std::vector<int>& inlier_indices1, std::vector<char>& mask)
{
  std::fill(mask.begin(), mask.end(), false);
  std::unordered_set<int> inlier_set1(inlier_indices1.begin(), inlier_indices1.end());

  for (int i = 0; i < matches_12.size(); ++i) {
    const cv::DMatch& dm = matches_12.at(i);
    if (inlier_set1.count(dm.queryIdx) > 0) {
      mask.at(i) = true;
    }
  }
}


void DrawLineMatches(const cv::Mat& img1,
                     const std::vector<ld::KeyLine>& keylines1,
                     const cv::Mat& img2,
                     const std::vector<ld::KeyLine>& keylines2,
                     const std::vector<cv::DMatch>& matches_12,
                     cv::Mat& draw_img,
                     const std::vector<char>& matches_mask,
                     bool draw_connectors)
{
  CV_Assert(img1.type() == img2.type());

  const int total_rows = img1.rows >= img2.rows ? img1.rows : img2.rows;
  draw_img = cv::Mat::zeros(total_rows, img1.cols + img2.cols, img1.type());

  cv::Mat roi_left(draw_img, cv::Rect(0, 0, img1.cols, img1.rows));
  cv::Mat roi_right(draw_img, cv::Rect(img1.cols, 0, img2.cols, img2.rows));
  img1.copyTo(roi_left);
  img2.copyTo(roi_right);

  const int offset = img1.cols;

  for (size_t counter = 0; counter < matches_12.size(); counter++) {
    if (matches_mask.size() == 0 || matches_mask.at(counter) != 0) {
      const cv::DMatch& dm = matches_12.at(counter);
      const ld::KeyLine& left = keylines1.at(dm.queryIdx);
      const ld::KeyLine& right = keylines2.at(dm.trainIdx);

      // Generates a random color of uniform brightness.
      const cv::Vec3b& random_color = viz::ToCvColor(core::RandomUnit3f());

      cv::line(draw_img,
          cv::Point2f(left.sPointInOctaveX, left.sPointInOctaveY),
          cv::Point2f(left.ePointInOctaveX, left.ePointInOctaveY),
          random_color, 2);

      cv::line(draw_img,
          cv::Point2f(right.sPointInOctaveX + offset, right.sPointInOctaveY),
          cv::Point2f(right.ePointInOctaveX + offset, right.ePointInOctaveY),
          random_color, 2);

      // Optionally draw associations.
      if (draw_connectors) {
        cv::line(draw_img,
        cv::Point2f(left.sPointInOctaveX, left.sPointInOctaveY),
        cv::Point2f(right.sPointInOctaveX + offset, right.sPointInOctaveY),
        random_color, 1);
      }
    }
  }
}

}
}
