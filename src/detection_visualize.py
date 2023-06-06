#!/usr/bin/env python3
import cv2
import message_filters
import norfair
import rospy
import numpy as np
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
        # Load parameters
        detection_visualizer = rospy.get_param("detection_visualizer")
        self.image_topic = detection_visualizer["image"]["topic"]
        self.detection_topic = detection_visualizer["detection"]["topic"]

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.detections_sub = message_filters.Subscriber(self.detection_topic, BoundingBoxArray)
        self.image_pub = rospy.Publisher("detection_image", Image, queue_size=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detections_sub], 10, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

        w, h = detection_visualizer["image"]["width"], detection_visualizer["image"]["height"]
        self.projecter = PixelCoordinatesProjecter([w, h], focal_length_pixel=np.array([502.7643127441406,502.8880310058594]),
                                                   principal_point_pixel=np.array([508.549560546875, 519.2774658203125]))
        q = detection_visualizer["tf_between_topics"]["q"]
        t = detection_visualizer["tf_between_topics"]["t"]
        self.R = q_to_R(q)
        self.t = np.array(t)

        rospy.init_node("detection_image")
        rospy.spin()

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

if __name__ == "__main__":
    detection_image = DetectionImage()