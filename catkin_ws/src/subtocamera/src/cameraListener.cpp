#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"

void chatterCallback(const sensor_msgs::Image::ConstPtr& image)
{
  ROS_INFO("I heard something");
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cameralistener");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/apollo/sensor/camera/traffic/image_center", 1000, chatterCallback);
  ros::spin();
  return 0;
}

