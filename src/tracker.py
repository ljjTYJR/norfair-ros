#!/usr/bin/env python3
"""
Use the norfair tracker to track objects by their bounding boxes.
"""
import norfair
import numpy as np
import rospy
import tf2_ros
from norfair import Detection, Tracker
from norfair_ros.msg import Point
from norfair_ros.msg import Detections as DetectionsMsg
from darko_perception_msgs.msg import SceneObjects, SceneObject
from utils import scaled_euclidean, get_pose_and_extents_from_points
import tf2_geometry_msgs

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
        subscribed_topic = rospy.get_param("tracker")["bounding_box_topic"]
        published_topic = rospy.get_param("published_topics")["tracked_scene_objects"]
        rospy.Subscriber(subscribed_topic, DetectionsMsg, self.pipeline)
        self.pub = rospy.Publisher(published_topic, SceneObjects, queue_size=10)
        self.map_frame = rospy.get_param("frame_setting")["map_frame"]
        self.detection_frame = rospy.get_param("frame_setting")["detection_frame"]
        self.tf_buffer = tf2_ros.Buffer()

        rospy.spin()

    def get_transform_between_frames(self, src: str, tgt: str) -> np.ndarray:
        """
        Get the transform between two frames.

        Parameters
        ----------
        src : str
            Source frame.
        tgt : str
            Target frame.

        Returns
        -------
        np.ndarray
            Transform between the two frames.
        """
        try:
            # TODO: The time should be from the message or now?
            transform = self.tf_buffer.lookup_transform(src, tgt, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr(f"Cannot get transform from {src} to {tgt}.")
            return None

        return transform

    def pipeline(self, detections_msg: DetectionsMsg) -> None:
        detections = []
        for detection in detections_msg.detections:
            detections.append(
                Detection(
                    points = np.array([point.point for point in detection.points]),
                )
            )
        tracked_objects = self.tracker.update(detections)
        objects = []

        for tracked_object in tracked_objects:
            points = [point for point in tracked_object.estimate]
            pose_, extents_ = get_pose_and_extents_from_points(points)

            transform = self.get_transform_between_frames(self.map_frame, self.detection_frame)
            if transform is not None:
                pose_ = tf2_geometry_msgs.do_transform_pose(pose_, transform)
            else:
                rospy.logerr("Cannot get transform from map to detection frame.")

            objects.append(
                SceneObject(
                    pose = pose_,
                    extents = extents_,
                )
            )
        scene_objects_msg = SceneObjects(objects = objects)

        self.pub.publish(scene_objects_msg)

if __name__ == "__main__":
    try:
        tracker = Tracker()
    except rospy.ROSInterruptException:
        pass