#pragma once

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"

#include <opencv2/line_descriptor/descriptor.hpp>

namespace ld = cv::line_descriptor;

namespace bm {
namespace vo {


class LineDetector final {
 public:
  struct Options {
    int lsd_num_features    = 50;  // Set this to -1 if you want ALL of the features.
    int lsd_scale           = 0;    // TODO(milo): Figure out what this is...
    int lsd_num_octaves     = 1;    // This only seems to work when set to 1.
  };

  LineDetector(const Options& opt) : opt_(opt) {}

  int Detect(const core::Image1b& img, std::vector<ld::KeyLine>& lines_out, cv::Mat& desc_out);

 private:
  Options opt_;
  cv::Ptr<ld::LSDDetector> lsd_ = ld::LSDDetector::createLSDDetector();
  cv::Ptr<ld::BinaryDescriptor> lbd_ = ld::BinaryDescriptor::createBinaryDescriptor();
};

}
}
