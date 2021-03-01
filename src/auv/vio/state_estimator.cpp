#include <glog/logging.h>

#include "vio/state_estimator.hpp"


namespace bm {
namespace vio {


StateEstimator::StateEstimator(const Options& opt)
    : opt_(opt),
      sf_in_queue_(opt_.max_queue_size_stereo, true),
      sf_out_queue_(opt_.max_queue_size_stereo, true),
      is_shutdown_(false)
{
  stereo_frontend_thread_ = std::thread(&StateEstimator::StereoFrontendLoop, this);
}


void StateEstimator::ReceiveStereo(const StereoImage& stereo_pair)
{
  sf_in_queue_.Push(stereo_pair);
}


void StateEstimator::StereoFrontendLoop()
{
  LOG(INFO) << "Started up StereoFrontendLoop() thread" << std::endl;
  while (!is_shutdown_) {
    // If no images waiting to be processed, take a nap.
    while (sf_in_queue_.Empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const StereoImage& stereo_pair = sf_in_queue_.Pop();
    LOG(INFO) << "Received stereo image " << stereo_pair.camera_id << std::endl;
  }
}

}
}
