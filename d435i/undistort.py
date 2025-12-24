#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def undistort_and_publish():
    
    rospy.init_node('image_undistort_publisher', anonymous=True)
   
    image_pub_1 = rospy.Publisher("/camera_1/color/image_rect", Image, queue_size=1)
    image_pub_2 = rospy.Publisher("/camera_2/color/image_rect", Image, queue_size=1)
  
    bridge = CvBridge()

    K1 = np.array([[613.524342, 0, 428.816115],
                   [0, 612.102576, 252.542034],
                   [0, 0, 1]])
    D1 = np.array([0.121403, -0.243338, 0.000430, 0.003692, 0.000000])  

    K2 = np.array([[623.61874, 0, 426.51426],
                   [0, 614.57412, 255.43817],
                   [0, 0, 1]])
    D2 = np.array([0.137203, -0.256427, 0.000372, 0.004192, 0.000000]) 

    width = 848  
    height = 480  

    image_sub_1 = rospy.Subscriber('/cam_1/color/image_raw', Image, image_callback,
                                   (image_pub_1, bridge, K1, D1, width, height))
    image_sub_2 = rospy.Subscriber('/cam_2/color/image_raw', Image, image_callback,
                                   (image_pub_2, bridge, K2, D2, width, height))
    rospy.spin()

def image_callback(msg, args):
    image_pub, bridge, K, D, width, height = args
 
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
   
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 1, (width, height))

    undistorted_image = cv2.undistort(cv_image, K, D, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y + h, x:x + w]

    undistorted_image = cv2.resize(undistorted_image, (width, height), interpolation=cv2.INTER_LINEAR)

    image_msg = bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8")
    image_msg.header.frame_id = "camera_color_optical_frame"  
    image_msg.header.stamp = rospy.Time.now() 

    image_pub.publish(image_msg)

if __name__ == '__main__':
    try:
        undistort_and_publish()
    except rospy.ROSInterruptException:
        pass
