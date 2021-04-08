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
                           std::vector<float>& error,
                           bool bidirectional,
                           float fwd_bkw_thresh_px)
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
                           cv::OPTFLOW_USE_INITIAL_FLOW & cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
                           0.001);

  if (bidirectional) {
    VecPoint2f px_ref_bkw;
    cv::calcOpticalFlowPyrLK(cur_img,
                            ref_img,
                            px_cur,
                            px_ref_bkw,
                            status,
                            error,
                            klt_window_size,
                            params_.klt_max_level,
                            kTerminationCriteria,
                            cv::OPTFLOW_USE_INITIAL_FLOW & cv::OPTFLOW_LK_GET_MIN_EIGENVALS,
                            0.001);

    // Invalidate any points that could be tracked in reverse.
    for (size_t i = 0; i < px_ref.size(); ++i) {
      const cv::Point2f& ref = px_ref.at(i);
      const cv::Point2f& ref_bkw = px_ref_bkw.at(i);
      const float dx = ref.x - ref_bkw.x;
      const float dy = ref.y - ref_bkw.y;
      if ((dx*dx + dy*dy) > fwd_bkw_thresh_px*fwd_bkw_thresh_px) {
        status.at(i) = 0;
      }
    }
  }

  // Invalidate any points that have tracked out of the image.
  for (size_t i = 0; i < px_cur.size(); ++i) {
    const cv::Point2f& pt = px_cur.at(i);
    if (pt.x <= 0 || pt.x >= cur_img.cols || pt.y <= 0 || pt.y >= cur_img.rows) {
      status.at(i) = 0;
    }
  }
}

}
}
