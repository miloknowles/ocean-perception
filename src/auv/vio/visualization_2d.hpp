#pragma once

#include <vector>

#include "core/cv_types.hpp"

namespace bm {
namespace vio {

using namespace core;


// Draws detected keypoints in a single image (no tracks).
Image3b DrawFeatures(const Image1b& cur_img, const std::vector<cv::KeyPoint>& cur_keypoints);


// Draws tracked keypoints with green arrows.
// Draws untracked keypoints in the previous frame in red.
// Draws newly initialized keypoints in the current frame in blue.
Image3b DrawFeatureTracks(const Image1b& ref_img,
                          const Image1b& cur_img,
                          const std::vector<cv::KeyPoint>& ref_keypoints,
                          const std::vector<cv::KeyPoint>& cur_keypoints,
                          const std::vector<cv::KeyPoint>& untracked_ref,
                          const std::vector<cv::KeyPoint>& untracked_cur);


}
}
