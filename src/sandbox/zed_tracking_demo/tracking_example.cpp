/*************************************************************************
 ** This sample demonstrates how to use the ZED for positional tracking  **
 ** and display camera motion in an OpenGL window. 		                   **
 **************************************************************************/

#include <sl/Camera.hpp>
#include "GLViewer.hpp"

// Using std namespace
using namespace std;
// using namespace sl;

#define IMU_ONLY 0
const int MAX_CHAR = 128;


inline void setTxt(sl::float3 value, char* ptr_txt)
{
  snprintf(ptr_txt, MAX_CHAR, "%3.2f; %3.2f; %3.2f", value.x, value.y, value.z);
}


void parseArgs(int argc, char **argv, sl::InitParameters& param);


int main(int argc, char **argv)
{
  sl::Camera zed;

  // Set configuration parameters for the ZED
  sl::InitParameters init_parameters;
  init_parameters.coordinate_units = sl::UNIT::METER;
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE; // RDF
  init_parameters.sdk_verbose = true;
  parseArgs(argc, argv, init_parameters);

  // Open the camera
  sl::ERROR_CODE returned_state = zed.open(init_parameters);
  if (returned_state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Camera returned status: " << returned_state << std::endl;
    return EXIT_FAILURE;
  }

  const sl::MODEL camera_model = zed.getCameraInformation().camera_model;
  GLViewer viewer;
  // Initialize OpenGL viewer
  viewer.init(argc, argv, camera_model);

  // Create text for GUI
  char text_rotation[MAX_CHAR];
  char text_translation[MAX_CHAR];

  // Set parameters for Positional Tracking
  sl::PositionalTrackingParameters positional_tracking_param;
  positional_tracking_param.enable_area_memory = true;
  // enable Positional Tracking
  returned_state = zed.enablePositionalTracking(positional_tracking_param);
  if (returned_state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Couldn't enable position tracking: " << returned_state << std::endl;
    zed.close();
    return EXIT_FAILURE;
  }

  sl::Pose world_P_cam;
  sl::POSITIONAL_TRACKING_STATE tracking_state;
#if IMU_ONLY
  sl::SensorsData sensors_data;
#endif

  while (viewer.isAvailable()) {
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
      // Get the position of the camera in a fixed reference frame (the World Frame)
      tracking_state = zed.getPosition(world_P_cam, sl::REFERENCE_FRAME::WORLD);

#if IMU_ONLY
      if (zed.getSensorsData(sensors_data, TIME_REFERENCE::IMAGE) == sl::ERROR_CODE::SUCCESS) {
        setTxt(sensors_data.imu.pose.getEulerAngles(), text_rotation); //only rotation is computed for IMU
        viewer.updateData(sensors_data.imu.pose, string(text_translation), string(text_rotation), sl::POSITIONAL_TRACKING_STATE::OK);
      }
#else
      if (tracking_state == sl::POSITIONAL_TRACKING_STATE::OK) {
        // Get rotation and translation and displays it
        setTxt(world_P_cam.getEulerAngles(), text_rotation);
        setTxt(world_P_cam.getTranslation(), text_translation);
      }

      // Update rotation, translation and tracking state values in the OpenGL window
      viewer.updateData(world_P_cam.pose_data, string(text_translation), string(text_rotation), tracking_state);
#endif

    } else
      sl::sleep_ms(1);
  }

  zed.disablePositionalTracking();

  //zed.disableRecording();
  zed.close();
  return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, sl::InitParameters& param)
{
  if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
    // SVO input mode
    param.input.setFromSVOFile(argv[1]);
    cout << "[Sample] Using SVO File input: " << argv[1] << endl;
  } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
    string arg = string(argv[1]);
    unsigned int a, b, c, d, port;
    if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
      // Stream input mode - IP + port
      string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
      param.input.setFromStream(sl::String(ip_adress.c_str()), port);
      cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
    } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
      // Stream input mode - IP only
      param.input.setFromStream(sl::String(argv[1]));
      cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
    } else if (arg.find("HD2K") != string::npos) {
      param.camera_resolution = sl::RESOLUTION::HD2K;
      cout << "[Sample] Using Camera in resolution HD2K" << endl;
    } else if (arg.find("HD1080") != string::npos) {
      param.camera_resolution = sl::RESOLUTION::HD1080;
      cout << "[Sample] Using Camera in resolution HD1080" << endl;
    } else if (arg.find("HD720") != string::npos) {
      param.camera_resolution = sl::RESOLUTION::HD720;
      cout << "[Sample] Using Camera in resolution HD720" << endl;
    } else if (arg.find("VGA") != string::npos) {
      param.camera_resolution = sl::RESOLUTION::VGA;
      cout << "[Sample] Using Camera in resolution VGA" << endl;
    }
  } else {
    // Default
  }
}
