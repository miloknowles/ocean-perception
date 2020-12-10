#pragma once

#include "core/eigen_types.hpp"
#include "core/cv_types.hpp"
#include "line_descriptor/include/line_descriptor_custom.hpp"

namespace bm {
namespace vo {

namespace ld = cv::line_descriptor;


class LineDetector final {
 public:
  struct Options {
    int lsd_num_features    = 300;  // Set this to -1 if you want ALL of the features.
    int lsd_refine          = 0;
    double lsd_scale        = 1.2;
    double lsd_sigma_scale  = 0.6;
    double lsd_quant        = 2.0;
    double lsd_ang_th       = 22.5;
    double log_eps          = 1.0;
    double lsd_density_th   = 0.6;
    int lsd_n_bins          = 1024;
    double lsd_min_length   = 20;   // TODO
  };

  LineDetector(const Options& opt) : opt_(opt) {
    lsd_opt_.refine = opt.lsd_refine;
    lsd_opt_.scale = opt.lsd_scale;
    lsd_opt_.sigma_scale = opt.lsd_sigma_scale;
    lsd_opt_.quant = opt.lsd_quant;
    lsd_opt_.ang_th = opt.lsd_ang_th;
    lsd_opt_.density_th = opt.lsd_density_th;
    lsd_opt_.n_bins = opt.lsd_n_bins;
    lsd_opt_.min_length = opt.lsd_min_length;
  }

  int Detect(const core::Image1b& img, std::vector<ld::KeyLine>& lines_out, cv::Mat& desc_out);

 private:
  Options opt_;
  ld::LSDDetectorC::LSDOptions lsd_opt_;

  cv::Ptr<ld::LSDDetectorC> lsd_ = ld::LSDDetectorC::createLSDDetectorC();
  cv::Ptr<ld::BinaryDescriptor> lbd_ = ld::BinaryDescriptor::createBinaryDescriptor();
};

}
}
