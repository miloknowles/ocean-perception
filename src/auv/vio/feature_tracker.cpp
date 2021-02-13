#include <glog/logging.h>

#include <opencv2/features2d/features2d.hpp>
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
  px_cur.clear();
  status.clear();
  error.clear();

  if (px_ref.empty()) {
    LOG(WARNING) << "No keypoints in reference frame!" << std::endl;
    return;
  }

  // std::vector<uchar> status;
  // std::vector<float> error;

  // Setup termination criteria for optical flow.
  const cv::TermCriteria kTerminationCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
      opt_.klt_maxiters,
      opt_.klt_epsilon);

  const cv::Size2i klt_window_size(opt_.klt_winsize, opt_.klt_winsize);

  // Initialize px_cur to their previous locations.
  // TODO(milo): Use motion to predict keypoint locations.
  px_cur = px_ref;

  cv::calcOpticalFlowPyrLK(ref_img,
                           cur_img,
                           px_ref,
                           px_cur,
                           status,
                           error,
                           klt_window_size,
                           opt_.klt_max_level,
                           kTerminationCriteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
}

}
}
