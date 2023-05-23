#!/usr/bin/env python3
"""
subscribe the
    Image topic: "/robot/k4a_bottom/rgb_to_depth/image_raw"
    Detections topic: "/robot/k4a_bottom/rgbd_yolo_9dof_ros/bboxes3d"
and project the 3D bounding boxes to 2D image plane, then publish the 2D bounding boxes to
    Output topic: "detection_image"
"""
import cv2
import message_filters
import norfair
import numpy as np
import rospy
from cv_bridge import CvBridge
from norfair.drawing import Drawable
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
from utils import PixelCoordinatesProjecter, get_points_from_boudning_box_msg, draw_3d_tracked_boxes

def q_to_R(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R

def align_image_and_bbox_coordinates(points, R, t):
    points = np.dot(R, points.T).T + t
    return points

class DetectionImage:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/robot/k4a_bottom/rgb_to_depth/image_raw"
        self.detection_topic = "/robot/k4a_bottom/rgbd_yolo_9dof_ros/bboxes3d"
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.detections_sub = message_filters.Subscriber(self.detection_topic, BoundingBoxArray)
        self.image_pub = rospy.Publisher("detection_image", Image, queue_size=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detections_sub], 10, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.callback)
        #TODO: These are magic numbers, the image size is 1024x1024
        # The R and t are got from `tf_echo`
        self.projecter = PixelCoordinatesProjecter([1024,1024])
        self.R = q_to_R([0.522, 0.528, -0.471, 0.476])
        self.t = np.array([0.004, 0.004, -0.051])

    def callback(self, image, bbox_array):
        # The frame is 1024x1024
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        # filp the image
        frame = cv2.flip(frame, 1)
        bounding_boxes = []
        for bbox in bbox_array.boxes:
            points = get_points_from_boudning_box_msg(bbox)
            points = align_image_and_bbox_coordinates(points, self.R, self.t)
            bounding_boxes.append(points)
        frame = draw_3d_tracked_boxes(
            frame,
            bounding_boxes,
            projecter = self.projecter.eye_2_pixel,
        )
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def main(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("detection_image")
    detection_image = DetectionImage()
    detection_image.main()