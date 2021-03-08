#pragma once

#include <vector>

#include "core/params_base.hpp"
#include "core/macros.hpp"
#include "core/cv_types.hpp"

namespace bm {
namespace vio {

using namespace core;


class StereoMatcher final {
 public:
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    int templ_cols = 31;                // Width of patch
    int templ_rows = 11;                // Height of patch
    int max_disp = 128;                 // disp = fx * B / depth
    double max_matching_cost = 0.15;    // Maximum matching cost considered valid
    bool bidirectional = false;
    bool subpixel_refinement = false;

   private:
    void LoadParams(const YamlParser& parser) override
    {
      parser.GetYamlParam("templ_cols", &templ_cols);
      parser.GetYamlParam("templ_rows", &templ_rows);
      parser.GetYamlParam("max_disp", &max_disp);
      parser.GetYamlParam("max_matching_cost", &max_matching_cost);
      parser.GetYamlParam("bidirectional", &bidirectional);
      parser.GetYamlParam("subpixel_refinement", &subpixel_refinement);
    }
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(StereoMatcher);

  explicit StereoMatcher(const Params& params) : params_(params) {}

  // Uses template-matching to find a left_keypoint in the right image (horizontal search).
  double MatchRectified(const Image1b& left_rectified,
                        const Image1b& right_rectified,
                        const cv::Point2f& left_keypoint);

  // Match a set of keypoints in the left image (calls member function above).
  std::vector<double> MatchRectified(const Image1b& left_rectified,
                                     const Image1b& right_rectified,
                                     const VecPoint2f& left_keypoints);

 private:
  Params params_;
};

}
}
