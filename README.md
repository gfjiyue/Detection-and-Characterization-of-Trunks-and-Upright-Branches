# Detection-and-Characterization-of-Trunks-and-Upright-Branches
This is an implementation of our published paper. In this paper, we proposed an automatic method for real-time detection and pruning of pear tree trunks and upright branches using an RGB-D camera.

# Instruction
We developed a PRUNING_ROS package, which runs on Ubuntu 20.04 with the Robot Operating System (ROS-Noetic), for real-time detection and characterization of pear tree trunks and upright branches, along with automated pruning of upright branches. 

# Camera Distortion Removal
package: d435i

rosrun d435i undistort.py

# Improved YOLOv5 for trunk detection
package: yolov5-k-means-SE
(Refer to the Usage Methods of YOLOv5: https://docs.ultralytics.com/zh/models/yolov5/)

# Improved YOLOv8
package: yolov8-dsc-fs
(Refer to the Usage Methods of YOLOv8: https://docs.ultralytics.com/zh/models/yolov8/)

# Trunk Positioning Model
package: trunk

roslaunch yolov5_ros yoloros.launch

# Branch Pruning Model
package: prune

rosrun yolo_ros point.py

rosrun yolo_ros skeleton.py

rosrun yolo_ros angle.py

# Video
The video is an illustration of robotic arm application. See it in the main.
