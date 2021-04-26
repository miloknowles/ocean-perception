#pragma once

#include <vector>

#include "core/macros.hpp"
#include "params/params_base.hpp"
#include "vision_core/cv_types.hpp"

namespace bm {
namespace ft {

using namespace core;


class FeatureTracker final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    int klt_maxiters = 30;
    float klt_epsilon = 0.001;
    int klt_winsize = 21;
    int klt_max_level = 4;

   private:
    void LoadParams(const YamlParser& parser) override;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(FeatureTracker);

  FeatureTracker() = delete;

  // Construct with options.
  explicit FeatureTracker(const Params& params) : params_(params) {}

  // Track points from ref_img to cur_img using Lucas-Kanade optical flow.
  // If px_cur is provided, these locations are used as an initial guess for the flow.
  // Otherwise, points are tracked from their reference locations.
  void Track(const Image1b& ref_img,
             const Image1b& cur_img,
             const VecPoint2f& px_ref,
             VecPoint2f& px_cur,
             std::vector<uchar>& status,
             std::vector<float>& error,
             bool bidirectional = false,
             float fwd_bkw_thresh_px = 5.0);

 private:
  Params params_;
};


}
}
