#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


void ImageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  try {
    cv::Mat image = cv::imdecode(cv::Mat(msg->data),1);
    cv::imshow("image_fl", image);
    cv::waitKey(30);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not decode image");
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "debug_image_node");
  ros::NodeHandle nh;
  cv::namedWindow("image_fl");
  ros::Subscriber sub = nh.subscribe("/simulator/sensors/camera_fl/compressed", 1, ImageCallback);
  ros::spin();
}
