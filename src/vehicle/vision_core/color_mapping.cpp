#include <opencv2/imgproc.hpp>

#include "vision_core/color_mapping.hpp"

namespace bm {
namespace core {


std::vector<cv::Vec3b> ColormapVector(const std::vector<double>& values,
                                      double vmin, double vmax,
                                      int cv_colormap)
{
  assert(vmax > vmin);

  const double vrange = vmax - vmin;
  cv::Mat1b values_1b;
  cv::Mat3b values_3b;
  cv::Mat1d(values).reshape(0, values.size()).convertTo(
      values_1b, CV_8UC1, 255.0 / vrange, vmin);
  cv::applyColorMap(values_1b, values_3b, cv_colormap);

  std::vector<cv::Vec3b> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out.at(i) = values_3b.at<cv::Vec3b>(i);
  }

  return out;
}

}
}
