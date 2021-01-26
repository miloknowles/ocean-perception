#include <opencv2/imgproc.hpp>

#include "imaging/fast_guided_filter.hpp"
#include "imaging/illuminant.hpp"

namespace bm {
namespace imaging {


Image3f EstimateIlluminantGaussian(const Image3f& bgr,
                                   int ksizeX,
                                   int ksizeY,
                                   double sigmaX,
                                   double sigmaY)
{
  Image3f lsac;
  cv::GaussianBlur(bgr, lsac, cv::Size(ksizeX, ksizeY), sigmaX, sigmaY, cv::BORDER_REPLICATE);

  // Akkaynak et al. multiply by a factor of 2 to get the illuminant map.
  return 2.0f * lsac;
}


Image3f EstimateIlluminantRangeGuided(const Image3f& bgr,
                                      const Image1f& range,
                                      int r,
                                      double eps,
                                      int s)
{
  const Image3f& lsac = fastGuidedFilter(range, bgr, r, eps, s);

  // Akkaynak et al. multiply by a factor of 2 to get the illuminant map.
  return 2.0f * lsac;
}


}
}
