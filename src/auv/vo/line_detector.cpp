#include "line_detector.hpp"

namespace bm {
namespace vo {


// From: https://github.com/rubengooj/stvo-pl
struct sort_lines_by_response {
  inline bool operator()(const ld::KeyLine& a, const ld::KeyLine& b) {
    return ( a.response > b.response );
  }
};


int LineDetector::Detect(const core::Image1b& img,
                         std::vector<ld::KeyLine>& lines_out,
                         cv::Mat& desc_out)
{
  lines_out.clear();

  // NOTE(milo): Not sure why members of opt are repeated here? Could it be a different param?
  lsd_->detect(img, lines_out, lsd_opt_.scale, 1, lsd_opt_);

  // If more lines than desired, filter out some of them.
  if (lines_out.size() > opt_.lsd_num_features && opt_.lsd_num_features > 0) {
      std::sort(lines_out.begin(), lines_out.end(), sort_lines_by_response());
      lines_out.resize(opt_.lsd_num_features);

      // Re-assign line indices after sorting.
      for (int i = 0; i < lines_out.size(); ++i) {
        lines_out.at(i).class_id = i;
      }
  }

  // Compute a visual descriptor for each line.
  lbd_->compute(img, lines_out, desc_out);

  return lines_out.size();
}

}
}
