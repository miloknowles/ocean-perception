#include <opencv2/highgui.hpp>
#include "imaging/io.hpp"

namespace bm {
namespace imaging {


Image1f LoadDepthTif(const std::string& filepath)
{
  return cv::imread(filepath, CV_LOAD_IMAGE_ANYDEPTH);
}


}
}
