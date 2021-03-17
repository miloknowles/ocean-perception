#include <glog/logging.h>
#include <opencv2/video/tracking.hpp>

#include "vio/feature_tracker.hpp"

namespace bm {
namespace vio {


void FeatureTracker::Track(const Image1b& ref_img,
                           const Image1b& cur_img,
                           const VecPoint2f& px_ref,
                           VecPoint2f& px_cur,
                           std::vector<uchar>& status,
                           std::vector<float>& error)
{
  status.clear();
  error.clear();

  if (px_ref.empty()) {
    LOG(WARNING) << "No keypoints in reference frame!" << std::endl;
    return;
  }

  // Setup termination criteria for optical flow.
  const cv::TermCriteria kTerminationCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
      params_.klt_maxiters,
      params_.klt_epsilon);

  const cv::Size2i klt_window_size(params_.klt_winsize, params_.klt_winsize);

  // If no initial guesses are provided for the optical flow, nitialize px_cur to previous locations.
  if (px_cur.empty()) {
    px_cur = px_ref;
  }

  cv::calcOpticalFlowPyrLK(ref_img,
                           cur_img,
                           px_ref,
                           px_cur,
                           status,
                           error,
                           klt_window_size,
                           params_.klt_max_level,
                           kTerminationCriteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
}

}
}
