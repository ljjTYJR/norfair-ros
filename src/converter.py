#!/usr/bin/env python3
import rospy
import numpy as np
from norfair_ros.msg import Detection as DetectionMsg
from norfair_ros.msg import Detections as DetectionsMsg
from norfair_ros.msg import Point
from darko_perception_msgs.msg import SceneObjects
from utils import pose_extents_to_keypoints

class Converter:
    """
    The Converter class is a ROS node that converts different input a standard bounding box message.
    """

    def scene_objects_to_norfair(self, scene_objects: SceneObjects) -> None:
        """
        Convert SceneObjects message to DetectionsMsg message.

        Parameters
        ----------
        scene_objects : SceneObjects
            SceneObjects message from `/perception/scene_objects`.
        """
        detections = []
        for so in scene_objects.objects:
            detections.append(
                DetectionMsg(
                    id = 0,
                    label = str(so.class_label),
                    points = [Point([p[0], p[1], p[2]]) for p in pose_extents_to_keypoints(so.pose.pose, so.extents)],
                    # scores=[so.confidence, so.confidence],
                    orientation = so.pose.pose.orientation,
                    position = so.pose.pose.position,
                    extents = so.extents,
                )
            )

        detections_msg = DetectionsMsg()
        detections_msg.header = scene_objects.header
        detections_msg.detections = detections

        self.converter_publisher.publish(detections_msg)

    def main(self) -> None:
        rospy.init_node("converter")

        subscribed_topic = rospy.get_param("subscribed_topics")["input_scene_objects"]
        published_topic = rospy.get_param("tracker")["bounding_box_topic"]

        rospy.Subscriber(subscribed_topic, SceneObjects, self.scene_objects_to_norfair)
        self.converter_publisher = rospy.Publisher(
            published_topic, DetectionsMsg, queue_size = 10
        )

        rospy.spin()

if __name__ == "__main__":
    try:
        Converter().main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Converter node terminated.")
