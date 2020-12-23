#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    print("I head something")
    #print(data)
    print(data.height)
    cv_image = image_converter.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    print(cv_image)
    #cv2.imshow("window", data.data)
    #cv.WaitKey(-1)
    
def listener():

    rospy.init_node('pycameralistener', anonymous=True)

    rospy.Subscriber('/apollo/sensor/camera/traffic/image_center', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
