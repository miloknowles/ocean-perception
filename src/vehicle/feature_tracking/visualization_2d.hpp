#pragma once

#include <vector>

#include "core/cv_types.hpp"

namespace bm {
namespace ft {

using namespace core;


// Draws detected keypoints in a single image (no tracks).
Image3b DrawFeatures(const Image1b& cur_img, const VecPoint2f& cur_keypoints);


// Draws tracked keypoints with green arrows.
// Draws untracked keypoints in the previous frame in red.
// Draws newly initialized keypoints in the current frame in blue.
Image3b DrawFeatureTracks(const Image1b& cur_img,
                          const VecPoint2f& ref_keypoints,
                          const VecPoint2f& cur_keypoints,
                          const VecPoint2f& untracked_ref,
                          const VecPoint2f& untracked_cur);


// Draws matched keypoints in green, and unmatched ones in red.
// Draws lines between corrspondences in the left/right images.
Image3b DrawStereoMatches(const Image1b& left,
                          const Image1b& right,
                          const VecPoint2f& keypoints_left,
                          const std::vector<double>& disp_left);


}
}
