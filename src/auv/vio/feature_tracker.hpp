#pragma once

#include <vector>

#include "core/macros.hpp"
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
    float klt_epsilon = 0.001;
    int klt_winsize = 21;
    int klt_max_level = 4;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(FeatureTracker);

  FeatureTracker() = delete;

  // Construct with options.
  explicit FeatureTracker(const Options& opt) : opt_(opt) {}

  // Track points from ref_img to cur_img using Lucas-Kanade optical flow.
  // If px_cur is provided, these locations are used as an initial guess for the flow.
  // Otherwise, points are tracked from their reference locations.
  void Track(const Image1b& ref_img,
             const Image1b& cur_img,
             const VecPoint2f& px_ref,
             VecPoint2f& px_cur,
             std::vector<uchar>& status,
             std::vector<float>& error);

 private:
  Options opt_;
};


}
}
