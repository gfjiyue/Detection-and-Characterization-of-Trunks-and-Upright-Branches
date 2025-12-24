#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
import rospy
import numpy as np
import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import Point

torch.backends.cudnn.enabled = False
class Yolo_Dect:
    def __init__(self):

        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/camera_1/color/image_rect')
        depth_topic = rospy.get_param('~depth_topic', '/camera_1/aligned_depth_to_color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.7')

        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')

        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = None
        self.depth_image = None
        self.getImageStatus = False
        self.classes_colors = {}
        self.prev_time = time.time()
        self.frame_count = 0
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1, buff_size=52428800)
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=1)
        self.center_pub = rospy.Publisher('/yolov5/center_points', Point, queue_size=10)

        while not self.getImageStatus:
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.getImageStatus = True
        self.process_images()

    def depth_callback(self, image):
        depth_array = np.frombuffer(image.data, dtype=np.uint16).astype(np.float32)
        expected_size = image.height * image.width
        if depth_array.size == expected_size:
            self.depth_image = depth_array.reshape((image.height, image.width))
        else:
            rospy.logerr(f"Depth image size mismatch: expected {expected_size}, got {depth_array.size}")

    def process_images(self):
        if self.color_image is not None and self.depth_image is not None:
            results = self.model(self.color_image)

            boxs = results.pandas().xyxy[0].values
            self.dectshow(self.color_image, self.depth_image, boxs)
            cv2.waitKey(3)

    def dectshow(self, color_image, depth_image, boxs):
        img = color_image.copy()
        height, width = img.shape[:2]
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = Header(stamp=rospy.Time.now(), frame_id=self.camera_frame)
        self.boundingBoxes.image_header = self.boundingBoxes.header

        self.frame_count += 1
        curr_time = time.time()
        elapsed_time = curr_time - self.prev_time
        if elapsed_time > 1.0:
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.prev_time = curr_time
        else:
            fps = self.frame_count / elapsed_time

        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        valid_boxes = []
        for box in boxs:
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                depth_value = depth_image[center_y, center_x] / 1000.0  
            else:
                depth_value = float('nan')

            if not np.isnan(depth_value) and depth_value < 3.0:
                valid_boxes.append((box, depth_value))

        for box, depth_value in valid_boxes:
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.Class = box[-1]
            boundingBox.depth = depth_value * 1000.0  

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (int(color[0]), int(color[1]), int(color[2])), 2)

            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            text = f'{boundingBox.Class} ({depth_value:.2f} m)'
            text_pos_y = box[1] + 30 if box[1] < 20 else box[1] - 10
            cv2.putText(img, text, (int(box[0]), int(text_pos_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            center_point = Point(x=center_x, y=center_y, z=depth_value * 1000.0)
            self.center_pub.publish(center_point)

            self.boundingBoxes.bounding_boxes.append(boundingBox)

        if valid_boxes:
            rospy.loginfo(f"detect{len(valid_boxes)} targets")
        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(img, height, width)
        cv2.imshow('YOLOv5', img)

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now(), frame_id=self.camera_frame)
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":
    main()