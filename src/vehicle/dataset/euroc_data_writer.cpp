#include <glog/logging.h>
#include <fstream>
#include <opencv2/highgui.hpp>

#include "core/file_utils.hpp"

#include "dataset/euroc_data_writer.hpp"

namespace bm {
namespace dataset {


EurocDataWriter::EurocDataWriter(const std::string& folder)
    : folder_(Join(folder, "mav0"))
{
  if (Exists(folder)) {
    LOG(WARNING) << "Dataset folder already exists, overwriting" << std::endl;
    rmdir(folder);
  }

  imu_folder_ = Join(folder_, "imu0");
  depth_folder_ = Join(folder_, "depth0");
  left_folder_ = Join(folder_, "cam0");
  right_folder_ = Join(folder_, "cam1");

  mkdir(folder);
  mkdir(folder_);
  mkdir(imu_folder_);
  mkdir(depth_folder_);
  mkdir(left_folder_);
  mkdir(right_folder_);
  mkdir(Join(left_folder_, "data"));
  mkdir(Join(right_folder_, "data"));
}


void EurocDataWriter::WriteImu(const ImuMeasurement& data)
{
  std::ofstream out;
  out.open(Join(imu_folder_, "data.csv"), std::ios_base::app);

  char buf[100];
  const int sz = std::snprintf(buf, 100, "%zu,%lf,%lf,%lf,%lf,%lf,%lf\n",
      data.timestamp, data.w.x(), data.w.y(), data.w.z(), data.a.x(), data.a.y(), data.a.z());
  CHECK(sz < 100) << "Buffer overflow! Need to allocate larger char[]" << std::endl;

  out << std::string(buf);
  out.close();
}


void EurocDataWriter::WriteStereo(const StereoImage3b& data)
{
  std::ofstream ofl, ofr;
  ofl.open(Join(left_folder_, "data.csv"), std::ios_base::app);
  ofr.open(Join(right_folder_, "data.csv"), std::ios_base::app);

  char buf[100];

  const std::string img_name = std::to_string(data.camera_id) + ".png";
  const std::string data_img_name = Join("data", img_name);

  // https://stackoverflow.com/questions/153890/printing-leading-0s-in-c
  const int sz = std::snprintf(buf, 100, "%zu,%s\n",data.timestamp, img_name);
  CHECK(sz < 100) << "Buffer overflow! Need to allocate larger char[]" << std::endl;

  ofl << std::string(buf);
  ofr << std::string(buf);
  ofl.close();
  ofr.close();

  cv::imwrite(Join(left_folder_, data_img_name), data.left_image);
  cv::imwrite(Join(right_folder_, data_img_name), data.right_image);
}


}
}
