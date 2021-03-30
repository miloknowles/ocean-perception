#include <glog/logging.h>

#include "zed_recorder.hpp"

namespace bm {
namespace zed {

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
  LOG(INFO) << "Starting capture loop" << std::endl;

  sl::Camera zed;

  // Set configuration parameters for the ZED.
  sl::InitParameters init_parameters;
  init_parameters.coordinate_units = sl::UNIT::METER;
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE; // RDF
  init_parameters.sdk_verbose = true;
  init_parameters.depth_mode = sl::DEPTH_MODE::NONE;

  sl::ERROR_CODE returned_state = zed.open(init_parameters);
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

  sl::TimestampHandler ts;

  // Retrieve sensors data during 5 seconds.
  auto start_time = std::chrono::high_resolution_clock::now();
  int count = 0;
  double elapsed_ms = 0;

  sl::SensorsData sensors_data;

  while (elapsed_ms < 5000 && !shutdown_) {
    // Depending on your camera model, different sensors are available.
    // They do not run at the same rate: therefore, to not miss any new samples we iterate as fast as possible
    // and compare timestamps to determine when a given sensor's data has been updated.
    // NOTE: There is no need to acquire images with grab(). getSensorsData runs in a separate internal capture thread.
    if (zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT) == sl::ERROR_CODE::SUCCESS) {
      // Check if a new IMU sample is available. IMU is the sensor with the highest update frequency.
      if (ts.isNew(sensors_data.imu)) {
        std::cout << "Sample " << count++ << "\n";
        std::cout << " - IMU:\n";
        std::cout << " \t Orientation: {" << sensors_data.imu.pose.getOrientation() << "}\n";
        std::cout << " \t Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]\n";
        std::cout << " \t Angular Velocity: {" << sensors_data.imu.angular_velocity << "} [deg/sec]\n";

        // Check if Magnetometer data has been updated.
        if (ts.isNew(sensors_data.magnetometer))
            std::cout << " - Magnetometer\n \t Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]\n";

        // Check if Barometer data has been updated.
        if (ts.isNew(sensors_data.barometer))
            std::cout << " - Barometer\n \t Atmospheric pressure:" << sensors_data.barometer.pressure << " [hPa]\n";
      }
    }

    // Compute the elapsed_ms time since the beginning of the main loop.
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
  }

  LOG(INFO) << "Finished collecting data" << std::endl;
  zed.close();
}


}
}
