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
    int lsd_num_features    = 300;  // Set this to -1 if you want ALL of the features.
    int lsd_refine          = 0;
    int lsd_scale           = 2;
    int lsd_num_octaves     = 1;
    double lsd_sigma_scale  = 0.6;
    double lsd_quant        = 2.0;
    double lsd_ang_th       = 22.5;
    double log_eps          = 1.0;
    double lsd_density_th   = 0.6;
    int lsd_n_bins          = 1024;
    double lsd_min_length   = 20;   // TODO
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
