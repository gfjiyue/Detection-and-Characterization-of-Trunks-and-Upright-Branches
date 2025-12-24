#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import cv2
from skimage.morphology import skeletonize
from sensor_msgs.msg import Image
import os

rospy.init_node("ultralytics_cut_skeletonization", anonymous=True)

skeleton_coordinates_pub = rospy.Publisher("/ultralytics/cut/skeleton_coordinates", Float32MultiArray, queue_size=5)

def callback_coordinates(coordinate_msg):
    try:
        coordinates = np.array(coordinate_msg.data).reshape((-1, 2))  
        if coordinates.shape[0] == 0:
            rospy.loginfo("No coordinates received.")
            return
        
        image_width = 848  
        image_height = 480  

        binary_image = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for coord in coordinates:
            u, v = coord.astype(int)  
            if 0 <= u < image_width and 0 <= v < image_height:  
                binary_image[v, u] = 255  

        skeleton = skeletonize(binary_image // 255)  

        skeleton_coords = np.argwhere(skeleton == 1)


        if len(skeleton_coords) > 0:
            for coord in skeleton_coords:
                rospy.loginfo(f"Skeleton Coordinate: ({coord[0]}, {coord[1]})")  

            skeleton_coord_msg = Float32MultiArray()
            skeleton_coord_msg.data = skeleton_coords.flatten().tolist()  
            skeleton_coordinates_pub.publish(skeleton_coord_msg)

            if not os.path.exists(output_txt_path): 
                with open(output_txt_path, 'w') as f:
                    for coord in skeleton_coords:
                        f.write(f"{coord[0]}, {coord[1]}\n") 

        else:
            rospy.loginfo("No skeleton coordinates found.")

    except Exception as e:
        rospy.logerr(f"Error in processing: {e}")

rospy.Subscriber("/ultralytics/cut/coordinates", Float32MultiArray, callback_coordinates)

rospy.spin()
