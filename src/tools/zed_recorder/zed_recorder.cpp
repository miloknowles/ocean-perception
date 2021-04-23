#include <opencv2/imgproc.hpp>

#include "zed_recorder.hpp"
#include "core/imu_measurement.hpp"
#include "core/depth_measurement.hpp"
#include "core/stereo_image.hpp"
#include "core/math_util.hpp"
#include "dataset/euroc_data_writer.hpp"

namespace bm {
namespace zed {

using namespace core;



// Mapping between MAT_TYPE and CV_TYPE
// https://github.com/stereolabs/zed-opencv/blob/master/cpp/src/main.cpp
int getOCVtype(sl::MAT_TYPE type) {
  int cv_type = -1;
  switch (type) {
    case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
    default: break;
  }
  return cv_type;
}

/**
* Conversion function between sl::Mat and cv::Mat
* https://github.com/stereolabs/zed-opencv/blob/master/cpp/src/main.cpp
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
  // cv::Mat and sl::Mat will share a single memory structure
  return cv::Mat(input.getHeight(),
                 input.getWidth(),
                 getOCVtype(input.getDataType()),
                 input.getPtr<sl::uchar1>(sl::MEM::CPU),
                 input.getStepBytes(sl::MEM::CPU));
}


// Function to display sensor parameters.
void printSensorConfiguration(sl::SensorParameters& sp)
{
  if (sp.isAvailable) {
    std::cout << "*****************************" << std::endl;
    std::cout << "Sensor Type: " << sp.type << std::endl;
    std::cout << "Max Rate: "    << sp.sampling_rate << sl::SENSORS_UNIT::HERTZ << std::endl;
    std::cout << "Range: ["      << sp.range << "] " << sp.sensor_unit << std::endl;
    std::cout << "Resolution: "  << sp.resolution << " " << sp.sensor_unit << std::endl;
    if (std::isfinite(sp.noise_density)) std::cout << "Noise Density: " << sp.noise_density <<" "<< sp.sensor_unit<<"/√Hz"<<std::endl;
    if (std::isfinite(sp.random_walk)) std::cout << "Random Walk: " << sp.random_walk <<" "<< sp.sensor_unit<<"/s/√Hz"<<std::endl;
  }
}


ZedRecorder::ZedRecorder(const std::string& output_folder)
  : output_folder_(output_folder), shutdown_(false)
{
  LOG(INFO) << "Constructed ZedRecorder" << std::endl;
  LOG(INFO) << "Will save data in EuRoC format to: " << output_folder_ << std::endl;
}


void ZedRecorder::Run(bool blocking)
{
  thread_ = std::thread(&ZedRecorder::CaptureLoop, this);

  if (blocking) {
    while (!shutdown_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
}


void ZedRecorder::Shutdown()
{
  shutdown_.store(true);

  if (thread_.joinable()) {
    thread_.join();
  }
}


void ZedRecorder::CaptureLoop()
{
  sl::Camera zed;

  // Set configuration parameters for the ZED.
  sl::InitParameters initp;
  initp.coordinate_units = sl::UNIT::METER;
  initp.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE; // RDF
  initp.sdk_verbose = true;
  initp.depth_mode = sl::DEPTH_MODE::NONE;
  initp.camera_resolution = sl::RESOLUTION::VGA;

  sl::ERROR_CODE returned_state = zed.open(initp);
  if (returned_state != sl::ERROR_CODE::SUCCESS) {
    LOG(FATAL) << "Could not open camera, returned status " << returned_state << std::endl;
  }

  // Check camera model.
  auto info = zed.getCameraInformation();
  sl::MODEL cam_model =info.camera_model;
  if (cam_model == sl::MODEL::ZED) {
    LOG(FATAL) << "This tutorial only works with ZED 2 and ZED-M cameras. ZED does not have additional sensors.\n"<< std::endl;
  }

  // Display camera information (model, serial number, firmware versions).
  std::cout << "Camera Model: " << cam_model << std::endl;
  std::cout << "Serial Number: " << info.serial_number << std::endl;
  std::cout << "Camera Firmware: " << info.camera_configuration.firmware_version << std::endl;
  std::cout << "Sensors Firmware: " << info.sensors_configuration.firmware_version << std::endl;

  std::cout << "\n\n** STEREO CAMERA CALIBRATION **" << std::endl;

  std::cout << "Left Intrinsics:" << std::endl;
  printf("  fx=%lf  \n  fy=%lf  \n  cx=%lf  \n  cy=%lf\n",
      info.calibration_parameters.left_cam.fx,
      info.calibration_parameters.left_cam.fy,
      info.calibration_parameters.left_cam.cx,
      info.calibration_parameters.left_cam.cy);

  std::cout << "Right Intrinsics:" << std::endl;
  printf("  fx=%lf  \n  fy=%lf  \n  cx=%lf  \n  cy=%lf\n",
      info.calibration_parameters.right_cam.fx,
      info.calibration_parameters.right_cam.fy,
      info.calibration_parameters.right_cam.cx,
      info.calibration_parameters.right_cam.cy);

  printf("Image Resolution:\n  height=%zu  \n  width=%zu\n", info.camera_resolution.height, info.camera_resolution.width);

  const sl::Transform lTr = info.calibration_parameters.stereo_transform;
  Matrix3d left_R_right;
  Vector3d left_t_right(lTr.tx, lTr.ty, lTr.tz);
  left_R_right << lTr.r00, lTr.r01, lTr.r02, lTr.r10, lTr.r11, lTr.r12, lTr.r20, lTr.r21, lTr.r22;

  std::cout << "Stereo Rig Extrinsics:" << std::endl;
  std::cout << "left_R_right:" << std::endl;
  std::cout << left_R_right << std::endl;
  std::cout << "left_t_right:" << std::endl;
  std::cout << left_t_right.transpose() << std::endl;

  std::cout << "\n" << std::endl;

  // Display sensors configuration (imu, barometer, magnetometer).
  printSensorConfiguration(info.sensors_configuration.accelerometer_parameters);
  printSensorConfiguration(info.sensors_configuration.gyroscope_parameters);
  printSensorConfiguration(info.sensors_configuration.magnetometer_parameters);
  printSensorConfiguration(info.sensors_configuration.barometer_parameters);

  sl::TimestampHandler ts;

  // Retrieve sensors data during 5 seconds.
  auto start_time = std::chrono::high_resolution_clock::now();
  double elapsed_sec = 0;

  sl::SensorsData sensors_data;

  dataset::EurocDataWriter writer(output_folder_);

  LOG(INFO) << "Recording in progress" << std::endl;

  while (elapsed_sec < max_duration_sec_ && !shutdown_) {
    // Depending on your camera model, different sensors are available.
    // They do not run at the same rate: therefore, to not miss any new samples we iterate as fast as possible
    // and compare timestamps to determine when a given sensor's data has been updated.
    // NOTE: There is no need to acquire images with grab(). getSensorsData runs in a separate internal capture thread.
    if (zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT) == sl::ERROR_CODE::SUCCESS) {
      // Check if a new IMU sample is available. IMU is the sensor with the highest update frequency.
      if (ts.isNew(sensors_data.imu)) {
        // std::cout << "Sample " << count++ << "\n";
        // std::cout << " - IMU:\n";
        // std::cout << " \t Orientation: {" << sensors_data.imu.pose.getOrientation() << "}\n";
        // std::cout << " \t Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]\n";
        // std::cout << " \t Angular Velocity: {" << sensors_data.imu.angular_velocity << "} [deg/sec]\n";

        const timestamp_t timestamp = sensors_data.imu.timestamp.getNanoseconds();

        if (imu_sampler_.ShouldSample(timestamp)) {
          const Vector3d angular_vel(DegToRad(sensors_data.imu.angular_velocity.x),
                                     DegToRad(sensors_data.imu.angular_velocity.y),
                                     DegToRad(sensors_data.imu.angular_velocity.y));
          const Vector3d linear_accel(sensors_data.imu.linear_acceleration.x,
                                      sensors_data.imu.linear_acceleration.y,
                                      sensors_data.imu.linear_acceleration.z);
          writer.WriteImu(ImuMeasurement(timestamp, angular_vel, linear_accel));
        }

        // Check if Magnetometer data has been updated.
        if (ts.isNew(sensors_data.magnetometer))
          std::cout << " - Magnetometer\n \t Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]\n";

        // Check if Barometer data has been updated.
        if (ts.isNew(sensors_data.barometer))
          std::cout << " - Barometer\n \t Atmospheric pressure:" << sensors_data.barometer.pressure << " [hPa]\n";
      }

      // NOTE(milo): ZED returns a BGRA image, so we need to get the first 3 channels only.
      if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        const timestamp_t timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        if (cam_sampler_.ShouldSample(timestamp)) {
          sl::Mat iml, imr;
          zed.retrieveImage(iml, sl::VIEW::LEFT, sl::MEM::CPU);
          zed.retrieveImage(imr, sl::VIEW::RIGHT, sl::MEM::CPU);
          const cv::Mat iml_cv = slMat2cvMat(iml);
          const cv::Mat imr_cv = slMat2cvMat(imr);
          Image3b iml_cv_bgr, imr_cv_bgr;
          cv::cvtColor(iml_cv, iml_cv_bgr, cv::COLOR_BGRA2BGR);
          cv::cvtColor(imr_cv, imr_cv_bgr, cv::COLOR_BGRA2BGR);
          const StereoImage3b stereo_pair(timestamp, camera_id_, iml_cv_bgr, imr_cv_bgr);
          writer.WriteStereo(stereo_pair);
          ++camera_id_;
        }
      }
    }

    elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
  }

  LOG(INFO) << "Finished collecting data" << std::endl;
  zed.close();
}


}
}
