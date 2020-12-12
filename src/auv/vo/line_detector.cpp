#include "vo/line_detector.hpp"

namespace bm {
namespace vo {


// From: https://github.com/rubengooj/stvo-pl
struct sort_lines_by_response {
  inline bool operator()(const ld::KeyLine& a, const ld::KeyLine& b) {
    return (a.response > b.response);
  }
};


// Filters out horizontal lines that will cause stereo matching problems.
static void FilterLines(const std::vector<ld::KeyLine>& kls,
                        double min_slope,
                        std::vector<ld::KeyLine>& kls_out)
{
  kls_out.clear();

  for (int i = 0; i < kls.size(); ++i) {
    const ld::KeyLine kl = kls.at(i);

    const double slope = std::fabs((kl.endPointY - kl.startPointY) / (kl.endPointX - kl.startPointX));
    if (slope < min_slope) {
      continue;
    }

    kls_out.emplace_back(kl);
  }
}


int LineDetector::Detect(const core::Image1b& img,
                         std::vector<ld::KeyLine>& lines_out,
                         cv::Mat& desc_out)
{
  lines_out.clear();

  std::vector<ld::KeyLine> lines_initial;

  // NOTE(milo): Not sure why members of opt are repeated here? Could it be a different param?
  // lsd_->detect(img, lines_initial, lsd_opt_.scale, 1, lsd_opt_);
  lsd_->detect(img, lines_initial, opt_.lsd_scale, opt_.lsd_num_octaves);

  // Remove horizontal lines.
  FilterLines(lines_initial, 1e-2, lines_out);

  // If more lines than desired, filter out some of them.
  if (lines_out.size() > opt_.lsd_num_features && opt_.lsd_num_features > 0) {
    std::sort(lines_out.begin(), lines_out.end(), sort_lines_by_response());
    lines_out.resize(opt_.lsd_num_features);
  }

  // Re-assign line indices after sorting and filtering.
  for (int i = 0; i < lines_out.size(); ++i) {
    lines_out.at(i).class_id = i;
  }

  // Compute a visual descriptor for each line.
  lbd_->compute(img, lines_out, desc_out);

  return lines_out.size();
}

}
}
