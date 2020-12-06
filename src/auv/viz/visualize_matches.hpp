#pragma once

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "line_descriptor/include/line_descriptor_custom.hpp"

#include "viz/colormap.hpp"
#include "core/random.hpp"

namespace bm {
namespace viz {

namespace ld = cv::line_descriptor;


std::vector<cv::DMatch> ConvertToDMatch(const std::vector<int>& matches12);


void DrawLineMatches(const cv::Mat& img1,
                     const std::vector<ld::KeyLine>& keylines1,
                     const cv::Mat& img2,
                     const std::vector<ld::KeyLine>& keylines2,
                     const std::vector<cv::DMatch>& matches1to2,
                     cv::Mat& draw_img,
                     const std::vector<char>& matches_mask,
                     bool draw_connectors = false);

}
}
