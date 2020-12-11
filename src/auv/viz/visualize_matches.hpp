#pragma once

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "viz/colormap.hpp"
#include "core/random.hpp"

namespace ld2 = cv::ld2;

namespace bm {
namespace viz {


std::vector<cv::DMatch> ConvertToDMatch(const std::vector<int>& matches_12);


void FillMask(const std::vector<cv::DMatch>& matches_12, const std::vector<int>& inlier_indices1, std::vector<char>& mask);


void DrawLineMatches(const cv::Mat& img1,
                     const std::vector<ld2::KeyLine>& keylines1,
                     const cv::Mat& img2,
                     const std::vector<ld2::KeyLine>& keylines2,
                     const std::vector<cv::DMatch>& matches1to2,
                     cv::Mat& draw_img,
                     const std::vector<char>& matches_mask,
                     bool draw_connectors = false);

}
}
