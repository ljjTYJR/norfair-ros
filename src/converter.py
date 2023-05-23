#!/usr/bin/env python3
import rospy
import numpy as np
from norfair_ros.msg import BoundingBoxes
from norfair_ros.msg import Detection as DetectionMsg
from norfair_ros.msg import Detections as DetectionsMsg
from norfair_ros.msg import Point
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox

class Converter:
    """
    The Converter class is a ROS node that converts different input messages to a norfair_ros input message.
    """

    def boundingboxes_to_norfair(self, bounding_boxes: BoundingBoxArray) -> None:
        """
        Convert SceneObjects message to DetectionsMsg message.

        Parameters
        ----------
        scene_objects : SceneObjects
            SceneObjects message from `/perception/scene_objects`.
        """
        detections = []
        for bb in bounding_boxes.boxes:
            detections.append(
                DetectionMsg(
                    id = 0,
                    label = str(bb.label),
                    points = centroid_to_bounding_box(bb.pose, bb.dimensions),
                    # scores=[so.confidence, so.confidence],
                    orientation = bb.pose.orientation,
                    dimensions = bb.dimensions,
                    position = bb.pose.position,
                )
            )

        detections_msg = DetectionsMsg()
        detections_msg.header = bounding_boxes.header
        detections_msg.detections = detections

        self.converter_publisher.publish(detections_msg)

    def main(self) -> None:
        rospy.init_node("converter")

        # Load parameters
        subscribers = rospy.get_param("converter_subscribers")
        publishers = rospy.get_param("converter_publishers")
        darko_detector = subscribers["jsk"]
        output = publishers["output"]

        # ROS subscriber definition
        rospy.Subscriber(darko_detector["topic"], BoundingBoxArray, self.boundingboxes_to_norfair)
        self.converter_publisher = rospy.Publisher(
            output["topic"], DetectionsMsg, queue_size=output["queue_size"]
        )

        rospy.spin()


if __name__ == "__main__":
    try:
        Converter().main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Converter node terminated.")
