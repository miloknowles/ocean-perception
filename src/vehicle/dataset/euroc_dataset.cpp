#include <fstream>

#include <glog/logging.h>

#include "dataset/euroc_dataset.hpp"
#include "core/file_utils.hpp"

namespace bm {
namespace dataset {


EurocDataset::EurocDataset(const std::string& toplevel_path) : DataProvider()
{
  const std::string mav0_path = Join(toplevel_path, "mav0");

  const std::string cam0_path = Join(mav0_path, "cam0");
  const std::string cam1_path = Join(mav0_path, "cam1");
  ParseStereo(cam0_path, cam1_path);

  const std::string& imu_csv = Join(mav0_path, "imu0/data.csv");
  if (Exists(imu_csv)) {
    ParseImu(imu_csv);
  } else {
    LOG(WARNING) << "[MISSING DATA] No IMU measurements found!" << std::endl;
  }

  const std::string& pose_txt = Join(mav0_path, "imu0_poses.txt");
  if (Exists(pose_txt)) {
    ParseGroundtruth(pose_txt);
  } else {
    LOG(WARNING) << "[MISSING DATA] No groundtruth poses found!" << std::endl;
  }

  const std::string& depth_csv = Join(mav0_path, "depth0/data.csv");
  if (Exists(depth_csv)) {
    ParseDepth(depth_csv);
  } else {
    LOG(WARNING) << "[MISSING DATA] No depth measurements found!" << std::endl;
  }

  const std::string& range_csv = Join(mav0_path, "aps0/data.csv");
  if (Exists(range_csv)) {
    ParseRange(range_csv);
  } else {
    LOG(WARNING) << "[MISSING DATA] No range measurements found!" << std::endl;
  }
}


// NOTE(milo): Adapted from KIMERA-VIO
void EurocDataset::ParseImu(const std::string& data_csv_path)
{
  // NOTE(milo): EuRoC IMU lines follow this format:
  // timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
  // a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
  std::ifstream fin(data_csv_path.c_str());
  CHECK(fin.is_open()) << "Could not open file: " << data_csv_path << std::endl;

  // Skip the first line, containing the header.
  std::string line;
  std::getline(fin, line);

  size_t count = 0;
  double max_norm_acc = 0;
  double max_norm_rot_rate = 0;
  timestamp_t prev_timestamp = 0;

  // Read/store imu measurements, line by line.
  while (std::getline(fin, line)) {
    timestamp_t timestamp = 0;
    Vector6d gyr_acc_data;
    for (size_t i = 0; i < (gyr_acc_data.size() + 1ul); ++i) {
      size_t idx = line.find_first_of(',');
      if (i == 0) {
        timestamp = std::stoll(line.substr(0, idx));
      } else {
        gyr_acc_data(i - 1) = std::stod(line.substr(0, idx));
      }
      line = line.substr(idx + 1);
    }
    CHECK_GT(timestamp, prev_timestamp) << "Euroc IMU data is not in chronological order!";

    const double norm_acc = gyr_acc_data.tail(3).norm();
    max_norm_acc = std::max(max_norm_acc, norm_acc);

    const double norm_rot_rate = gyr_acc_data.head(3).norm();
    max_norm_rot_rate = std::max(max_norm_rot_rate, norm_rot_rate);

    imu_data.emplace_back(ImuMeasurement(timestamp, gyr_acc_data.head(3), gyr_acc_data.tail(3)));

    prev_timestamp = timestamp;
    ++count;
  }

  fin.close();

  const timestamp_t total_time = imu_data.back().timestamp - imu_data.front().timestamp;

  LOG(INFO) << "IMU average (hz): "
           << (1e9 * static_cast<double>(count) / static_cast<double>(total_time)) << '\n'
           << "Maximum measured rotation rate (rad/s): " << max_norm_rot_rate << '\n'
           << "Maximum measured acceleration (m/s^2): " << max_norm_acc;
}


void EurocDataset::ParseStereo(const std::string& cam0_path, const std::string& cam1_path)
{
  std::vector<timestamp_t> left_stamps, right_stamps;
  std::vector<std::string> lf, rf;

  ParseImageFolder(cam0_path, left_stamps, lf);
  ParseImageFolder(cam1_path, right_stamps, rf);

  const size_t N = left_stamps.size();
  CHECK(right_stamps.size() == N && lf.size() == N && rf.size() == N) <<
      "Different number of left/right images and timestamps" << std::endl;

  for (size_t i = 0; i < N; ++i) {
    CHECK(left_stamps.at(i) == right_stamps.at(i)) << "Left/right timestamps don't match!\n";
    CHECK(Exists(lf.at(i)));
    CHECK(Exists(rf.at(i)));

    stereo_data.emplace_back(StereoDatasetItem(left_stamps.at(i), lf.at(i), rf.at(i)));
  }
}


// NOTE(milo): Adapted from KIMERA-VIO
void EurocDataset::ParseImageFolder(const std::string& cam_folder,
                                    std::vector<timestamp_t>& output_timestamps,
                                    std::vector<std::string>& output_filenames)
{
  CHECK(!cam_folder.empty());

  const std::string data_csv_path = Join(cam_folder, "data.csv");
  std::ifstream fin(data_csv_path.c_str());
  CHECK(fin.is_open()) << "Cannot open file: " << data_csv_path;

  // Skip the first line, containing the header.
  std::string item;
  std::getline(fin, item);

  // Read/store list of image names.
  while (std::getline(fin, item)) {
    size_t idx = item.find_first_of(',');
    const timestamp_t timestamp = std::stoll(item.substr(0, idx));
    const std::string image_filename = Join(cam_folder, "data/" + item.substr(0, idx) + ".png");

    // NOTE(KIMERA): Strangely, on mac, it does not work if we use: item.substr(idx + 1).
    output_timestamps.emplace_back(timestamp);
    output_filenames.emplace_back(image_filename);
  }

  fin.close();
}


void EurocDataset::ParseGroundtruth(const std::string& gt_path)
{
  // Read in groundtruth poses.
  CHECK(Exists(gt_path)) << "Groundtruth pose file does not exist: " << gt_path << std::endl;

  std::ifstream stream(gt_path.c_str());
  std::string line;

  while (std::getline(stream, line)) {
    std::stringstream iss(line);

    std::string ns, qw, qx, qy, qz, tx, ty, tz;
    std::getline(iss, ns, ',');
    std::getline(iss, qw, ',');
    std::getline(iss, qx, ',');
    std::getline(iss, qy, ',');
    std::getline(iss, qz, ',');
    std::getline(iss, tx, ',');
    std::getline(iss, ty, ',');
    std::getline(iss, tz, ',');

    const Quaterniond q(std::stod(qw), std::stod(qx), std::stod(qy), std::stod(qz));
    const Vector3d t(std::stod(tx), std::stod(ty), std::stod(tz));

    Matrix4d world_T_body = Matrix4d::Identity();
    world_T_body.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    world_T_body.block<3, 1>(0, 3) = t;

    const timestamp_t timestamp = std::stoull(ns);
    pose_data.emplace_back(GroundtruthItem(timestamp, world_T_body));
  }

  stream.close();
  LOG(INFO) << "Read in " << pose_data.size() << " groundtruth poses" << std::endl;
}


void EurocDataset::ParseDepth(const std::string& depth_csv_path)
{
  std::ifstream fin(depth_csv_path.c_str());
  CHECK(fin.is_open()) << "Could not open file: " << depth_csv_path << std::endl;

  // Skip the first line, containing the header.
  std::string line;
  std::getline(fin, line);

  double min_depth = std::numeric_limits<double>::max();
  double max_depth = std::numeric_limits<double>::min();
  timestamp_t prev_timestamp = 0;

  // Read/store imu measurements, line by line.
  while (std::getline(fin, line)) {
    int comma_idx = line.find_first_of(',');
    const timestamp_t timestamp = std::stoll(line.substr(0, comma_idx));

    line = line.substr(comma_idx + 1);
    comma_idx = line.find_first_of(',');
    const double depth = std::stod(line.substr(0, comma_idx));

    CHECK_GT(timestamp, prev_timestamp) << "EuRoC depth data is not in chronological order!";

    min_depth = std::min(min_depth, depth);
    max_depth = std::max(max_depth, depth);

    depth_data.emplace_back(DepthMeasurement(timestamp, depth));
    prev_timestamp = timestamp;
  }

  fin.close();

  LOG(INFO) << "Read in " << depth_data.size() << " DEPTH measurements.\n"
            << "  earliest=" << depth_data.front().timestamp
            << "  latest=" << depth_data.back().timestamp << "\n"
            << "  min=" << min_depth << " max=" << max_depth << std::endl;
}


void EurocDataset::ParseRange(const std::string& range_csv_path)
{
  std::ifstream fin(range_csv_path.c_str());
  CHECK(fin.is_open()) << "Could not open file: " << range_csv_path << std::endl;

  // Skip the first line, containing the header.
  std::string line;
  std::getline(fin, line);

  double min_range = std::numeric_limits<double>::max();
  double max_range = std::numeric_limits<double>::min();
  timestamp_t prev_timestamp = 0;

  // Read/store imu measurements, line by line.
  while (std::getline(fin, line)) {
    std::stringstream iss(line);

    std::string ns, r, tx, ty, tz;
    std::getline(iss, ns, ',');
    std::getline(iss, r, ',');
    std::getline(iss, tx, ',');
    std::getline(iss, ty, ',');
    std::getline(iss, tz, ',');

    const timestamp_t timestamp = std::stoll(ns);
    const double range = std::stod(r);
    const Vector3d t(std::stod(tx), std::stod(ty), std::stod(tz));

    CHECK_GT(timestamp, prev_timestamp) << "EuRoC range data is not in chronological order!";

    min_range = std::min(min_range, range);
    max_range = std::max(max_range, range);

    range_data.emplace_back(RangeMeasurement(timestamp, range, t));
    prev_timestamp = timestamp;
  }

  fin.close();

  LOG(INFO) << "Read in " << range_data.size() << " RANGE measurements.\n"
            << "  earliest=" << range_data.front().timestamp
            << "  latest=" << range_data.back().timestamp << "\n"
            << "  min=" << min_range << " max=" << max_range << std::endl;
}


}
}
