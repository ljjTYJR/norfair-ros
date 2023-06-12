#!/usr/bin/env python3
"""
Use the norfair tracker to track objects by their bounding boxes.
"""
import norfair
import numpy as np
import rospy
from norfair import Detection, Tracker
from jsk_recognition_msgs.msg import BoundingBoxArray
from norfair_ros.msg import BoundingBoxes as BoundingBoxesMsg
from norfair_ros.msg import BoundingBox as BoundingBoxMsg
from norfair_ros.msg import Point
from utils import get_points_from_boudning_box_msg, scaled_euclidean

class Tracker:
    def __init__(self) -> None:
        rospy.init_node("norfair_tracker")

        self.tracker = norfair.Tracker(
            distance_function = scaled_euclidean,
            distance_threshold = 0.3,
            hit_counter_max = 15,
            initialization_delay = 7.5,
            pointwise_hit_counter_max = 3,
            detection_threshold = 0.02,
            past_detections_length = 3,
        )

        # Load parameters
        subscribed_params = rospy.get_param("subscribed_topics")
        tracker = rospy.get_param("tracker")
        self.detection_topic = subscribed_params["detection_bounding_box"]["topic"]
        self.tracked_bounding_boxes_topic = tracker["output"]["topic"]
        rospy.Subscriber(self.detection_topic, BoundingBoxArray, self.pipeline)
        self.pub = rospy.Publisher(self.tracked_bounding_boxes_topic, BoundingBoxesMsg, queue_size=10)

        rospy.spin()

    def pipeline(self, bboxes: BoundingBoxArray) -> None:
        detections = []
        for bbox in bboxes.boxes:
            detections.append(
                Detection(
                    points=get_points_from_boudning_box_msg(bbox),
                )
            )
        tracked_objects = self.tracker.update(detections)
        # Get the points of the tracked objects and save in the bounding box message
        bounding_boxes_msg = BoundingBoxesMsg()
        bounding_boxes_msg.header = bboxes.header
        for tracked_object in tracked_objects:
            bounding_boxes_msg.bounding_boxes.append(
                BoundingBoxMsg(
                    points = [Point(point=point) for point in tracked_object.estimate],
                )
            )
        self.pub.publish(bounding_boxes_msg)

if __name__ == "__main__":
    try:
        tracker = Tracker()
    except rospy.ROSInterruptException:
        pass