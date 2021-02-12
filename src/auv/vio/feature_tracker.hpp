#pragma once

#include <vector>

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace vio {

using namespace core;


class FeatureTracker final {
 public:
  struct Options final {
    Options() = default;
    int klt_maxiters = 30;
    float klt_epsilon = 0.1;
    int klt_winsize = 24;
    int klt_max_level = 4;
  };

  // Construct with options.
  explicit FeatureTracker(const Options& opt) : opt_(opt) {}

  // Track points from ref_img to cur_img using Lucas-Kanade optical flow.
  void Track(const Image1b& ref_img,
             const Image1b& cur_img,
             const VecPoint2f& px_ref,
             const Matrix3d& R_ref_cur,
             VecPoint2f& px_cur);

 private:
  Options opt_;
};


}
}
