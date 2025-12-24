#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO
import cv2

rospy.init_node("ultralytics_cut_coordinate_publisher", anonymous=True)

segmentation_model = YOLO("/home/c302/yo/src/yolo_ros/best.pt")

coordinates_pub = rospy.Publisher("/ultralytics/cut/coordinates", Float32MultiArray, queue_size=5)

minimum_point_pub = rospy.Publisher("/ultralytics/cut/pruning_point", Float32MultiArray, queue_size=5)
K = np.array([[623.61874, 0, 426.51426],
               [0, 614.57412, 255.43817],
               [0, 0, 1]])

def callback(depth_msg):

    try:
        rgb_msg = rospy.wait_for_message("/camera_2/color/image_rect", Image)
        rgb_image = ros_numpy.numpify(rgb_msg)
        depth_image = ros_numpy.numpify(depth_msg)

        results = segmentation_model(rgb_image)

        min_distance = np.inf
        min_mask = None
        min_name = None
        min_point = None  

        for index, cls in enumerate(results[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = results[0].names[class_index]
            if name == "cut":  
                mask = results[0].masks.data.cpu().numpy()[index, :, :].astype(np.uint8)
                mask_resized = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))

                obj_depth = depth_image[mask_resized == 1]
                obj_depth = obj_depth[~np.isnan(obj_depth)] 

                if len(obj_depth) > 0:
                    avg_distance = np.mean(obj_depth)  
                    if avg_distance < min_distance: 
                        min_distance = avg_distance
                        min_mask = mask_resized
                        min_name = name
                        min_coordinates = np.argwhere(min_mask == 1)
                        min_point = min_coordinates[min_coordinates[:, 0].argmax()]  

        if min_mask is not None:
            coordinates = np.argwhere(min_mask == 1)
            coordinates_list = coordinates.tolist()

            #rospy.loginfo(f"Coordinates for '{min_name}' with minimum distance:")
            #rospy.loginfo(f"Total coordinates count: {len(coordinates_list)}")
            for coord in coordinates_list:
                rospy.loginfo(f"Coordinate: ({coord[1]}, {coord[0]})") 
            coord_msg = Float32MultiArray()
            coord_msg.data = [coord for sublist in coordinates_list for coord in sublist] 
            coordinates_pub.publish(coord_msg)

            if min_point is not None:
                pixel_x, pixel_y = min_point[1], min_point[0]
                depth = depth_image[pixel_y, pixel_x] 

                x = (pixel_x - K[0, 2]) * depth / K[0, 0]
                y = (pixel_y - K[1, 2]) * depth / K[1, 1]
                z = depth  

                #rospy.loginfo(f"Minimum point in space coordinates: ({x}, {y}, {z})")

                min_point_msg = Float32MultiArray()
                min_point_msg.data = [x, y, z]
                minimum_point_pub.publish(min_point_msg)

        else:
            rospy.loginfo("No 'cut' objects detected.")

    except Exception as e:
        rospy.logerr(f"Error in processing: {e}")

rospy.Subscriber("/camera_2/aligned_depth_to_color/image_raw", Image, callback)

rospy.spin()
