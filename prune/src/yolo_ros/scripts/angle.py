#!/usr/bin/env python
import rospy
import message_filters
import numpy as np
import math
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32
import cv2
from cv_bridge import CvBridge

class PixelAngleCalculator:
    def __init__(self):
        rospy.init_node("pixel_angle_calculator", anonymous=True)
        self.pixel_sub = rospy.Subscriber("/ultralytics/cut/skeleton_coordinates", Float32MultiArray, self.pixel_callback)
        self.depth_sub = message_filters.Subscriber("/camera_2/aligned_depth_to_color/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.depth_callback)
        self.bridge = CvBridge()
        self.depth_image = None 
        self.angle_pub = rospy.Publisher("/angle", Float32, queue_size=10) 

    def depth_callback(self, depth_msg):

        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    def pixel_callback(self, msg):
        if self.depth_image is None:
            rospy.logwarn("Waiting for depth image.")
            return

        points_3d = []
        data = msg.data 
        for i in range(0, len(data), 2):
            u, v = int(data[i]), int(data[i+1])  
            if 0 <= v < self.depth_image.shape[0] and 0 <= u < self.depth_image.shape[1]: 
                depth_values = []
                for dv in range(-1, 2):
                    for du in range(-1, 2):
                        new_v, new_u = v + dv, u + du
                        if 0 <= new_v < self.depth_image.shape[0] and 0 <= new_u < self.depth_image.shape[1]:
                            d = self.depth_image[new_v, new_u]
                            if d > 0:
                                depth_values.append(d)
                if depth_values:
                    d_avg = np.mean(depth_values) / 1000  
                    points_3d.append((v, u, d_avg))
                    rospy.loginfo(f"Pixel ({u}, {v}) -> Depth Avg: {d_avg:.3f} m")
                else:
                    rospy.logwarn(f"Invalid depth at ({u}, {v})")
            else:
                rospy.logwarn(f"Pixel ({u}, {v}) out of bounds")

        points_3d.sort(reverse=True, key=lambda p: p[0]) 

        if len(points_3d) < 2:
            rospy.logwarn("Not enough points for length calculation.")
            return

        v1, u1, _ = points_3d[0]  
        rospy.loginfo(f"vmax: ({u1}, {v1})")  

        total_length = 0.0
        target_point = None

        for j in range(1, len(points_3d)):
            v2, u2, _ = points_3d[j]
            dist = np.linalg.norm(np.array([v2 - v1, u2 - u1])) 
            total_length += dist
            if total_length >= 30:  
                target_point = (v2, u2)
                break

        if target_point is None:
            rospy.logwarn("No point found at 3cm distance.")
            return

        v_eps, u_eps = target_point  
        rospy.loginfo(f"pixel: ({u_eps}, {v_eps})")

        delta_v = abs(v_eps - v1)
        delta_u = abs(u_eps - u1)

        base_angle = math.degrees(math.atan2(delta_v, delta_u))  
        if u_eps < u1:  
            alpha = 180 - base_angle  
        else:
            alpha = base_angle

        rospy.loginfo(f"angel: {alpha:.2f}°")
        self.angle_pub.publish(Float32(alpha))

if __name__ == "__main__":
    try:
        calculator = PixelAngleCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
