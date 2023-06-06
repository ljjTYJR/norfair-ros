#!/usr/bin/env python3
"""
Receive the tracked bounding boxes and visualize them in rviz.
"""

import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from norfair_ros.msg import BoundingBoxes as BoundingBoxesMsg
from utils import PixelCoordinatesProjecter, get_points_from_boudning_box_msg, draw_3d_tracked_boxes
from cv_bridge import CvBridge

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

class TrackerImage:
    def __init__(self):
        rospy.init_node("tracker_image")

        # Load parameters
        tracker_visualizer = rospy.get_param("tracker_visualizer")
        self.tracked_bounding_boxes_topic = tracker_visualizer["tracked"]["topic"]
        self.image_topic = tracker_visualizer["image"]["topic"]
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.tracked_bounding_boxes_sub = message_filters.Subscriber(self.tracked_bounding_boxes_topic, BoundingBoxesMsg)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.tracked_bounding_boxes_sub], 10, 0.5, allow_headerless=True)
        self.ts.registerCallback(self.pipeline)
        self.image_pub = rospy.Publisher(tracker_visualizer["output"]["topic"], Image, queue_size=10)

        # 3D to 2d images
        w, h = tracker_visualizer["image"]["width"], tracker_visualizer["image"]["height"]
        # K: [502.7643127441406, 0.0, 508.549560546875,
        #       0.0, 502.8880310058594, 519.2774658203125,
        #       0.0, 0.0, 1.0]
        self.projecter = PixelCoordinatesProjecter([w, h], focal_length_pixel=np.array([502.7643127441406,502.8880310058594]),
                                                   principal_point_pixel=np.array([508.549560546875, 519.2774658203125]))
        q = tracker_visualizer["tf_between_topics"]["q"]
        t = tracker_visualizer["tf_between_topics"]["t"]
        self.R = q_to_R(q)
        self.t = np.array(t)

        self.bridge = CvBridge()

        rospy.spin()

    def pipeline(self, image, tracked_bounding_boxes):
        frame = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        # filp the image
        # frame = cv2.flip(frame, 1)
        bounding_boxes = []
        for bbox in tracked_bounding_boxes.bounding_boxes:
            # For the `BoundingBoxesMsg`, the received message directly contains the 3D points
            points = np.array([[points.point[0], points.point[1], points.point[2]] for points in bbox.points])
            # Align the 3D points with the image coordinates
            points = align_image_and_bbox_coordinates(points, self.R, self.t)
            bounding_boxes.append(points)
        # Draw the bounding boxes
        frame = draw_3d_tracked_boxes(
            frame,
            bounding_boxes,
            projecter = self.projecter.eye_2_pixel,
        )
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

if __name__ == "__main__":
    try :
        TrackerImage()
    except rospy.ROSInterruptException:
        print("ROS Terminated.")
        pass