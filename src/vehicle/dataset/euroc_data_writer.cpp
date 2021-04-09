#include <glog/logging.h>
#include <fstream>

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
}


void EurocDataWriter::WriteImu(const ImuMeasurement& data)
{
  std::ofstream out;
  out.open(Join(imu_folder_, "data.csv"), std::ios_base::app);

  char buf[100];
  const int sz = std::snprintf(buf, 100, "%zu,%lf,%lf,%lf,%lf,%lf,%lf\n",
      data.timestamp, data.w.x(), data.w.y(), data.w.z(), data.a.x(), data.a.y(), data.a.z());

  out << std::string(buf);
  out.close();
}


}
}
