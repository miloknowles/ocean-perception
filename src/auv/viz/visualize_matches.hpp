#pragma once

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace bm {
namespace viz {


inline std::vector<cv::DMatch> ConvertToDMatch(const std::vector<int>& matches12)
{
  std::vector<cv::DMatch> out;
  for (int i = 0; i < matches12.size(); ++i) {
    if (matches12.at(i) < 0) { continue; }
    out.emplace_back(i, matches12.at(i), 0);
  }

  return out;
}

}
}
