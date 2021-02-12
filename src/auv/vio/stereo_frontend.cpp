#include <glog/logging.h>

#include "vio/stereo_frontend.hpp"

namespace bm {
namespace vio {


StereoFrontend::StereoFrontend(const Options& opt)
    : opt_(opt),
      detector_(opt.detector_options),
      tracker_(opt.tracker_options),
      matcher_(opt.matcher_options)
{
  LOG(INFO) << "Constructed StereoFrontend!" << std::endl;
}


StereoFrontend::Result StereoFrontend::Track(const StereoImage& stereo_pair,
                                             const Matrix4d& T_prev_cur_prior)
{

}


}
}
