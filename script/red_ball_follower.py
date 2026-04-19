#!/usr/bin/env python3

import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class RedBallFollower:
    def __init__(self):
        rospy.init_node('red_ball_follower')

        self.bridge = CvBridge()

        self.cmd_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

        self.rgb_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        self.latest_depth = None
        self.latest_depth_encoding = None

        self.target_distance = 1.0
        self.linear_gain = 0.5
        self.angular_gain = 0.0025

        self.max_linear = 0.25
        self.max_angular = 0.8

        self.search_angular_speed = 0.3

        self.image_width = None
        self.image_height = None

        rospy.loginfo("Red ball follower started.")

    def depth_callback(self, msg):
        try:
            # Common encodings: 32FC1 or 16UC1
            if msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            self.latest_depth = depth_image
            self.latest_depth_encoding = msg.encoding

        except CvBridgeError as e:
            rospy.logerr("Depth CvBridge error: %s", str(e))

    def get_depth_at_pixel(self, x, y):
        if self.latest_depth is None:
            return None

        h, w = self.latest_depth.shape[:2]

        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        depth_value = self.latest_depth[y, x]

        if self.latest_depth_encoding == '16UC1':
            # millimeters -> meters
            distance = float(depth_value) / 1000.0
        else:
            distance = float(depth_value)

        if np.isnan(distance) or np.isinf(distance) or distance <= 0.0:
            return None

        return distance

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("RGB CvBridge error: %s", str(e))
            return

        self.image_height, self.image_width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV, so use two ranges
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()

        if len(contours) == 0:
            rospy.loginfo_throttle(2, "No red object found. Rotating to search...")
            twist.angular.z = self.search_angular_speed
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)

            cv2.imshow("Red Mask", mask)
            cv2.imshow("RGB View", frame)
            cv2.waitKey(1)
            return

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 300:
            rospy.loginfo_throttle(2, "Red object too small. Searching...")
            twist.angular.z = self.search_angular_speed
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)

            cv2.imshow("Red Mask", mask)
            cv2.imshow("RGB View", frame)
            cv2.waitKey(1)
            return

        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        cx, cy = int(x), int(y)

        cv2.circle(frame, (cx, cy), int(radius), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        distance = self.get_depth_at_pixel(cx, cy)

        image_center_x = self.image_width // 2
        error_x = cx - image_center_x

        twist.angular.z = -self.angular_gain * error_x
        twist.angular.z = max(min(twist.angular.z, self.max_angular), -self.max_angular)

        if distance is not None:
            distance_error = distance - self.target_distance
            twist.linear.x = self.linear_gain * distance_error
            twist.linear.x = max(min(twist.linear.x, self.max_linear), -self.max_linear)

            cv2.putText(
                frame,
                "Dist: {:.2f} m".format(distance),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            twist.linear.x = 0.0
            rospy.loginfo_throttle(2, "Red object found but no valid depth.")

        self.cmd_pub.publish(twist)

        cv2.line(frame, (image_center_x, 0), (image_center_x, self.image_height), (255, 255, 0), 2)

        cv2.imshow("Red Mask", mask)
        cv2.imshow("RGB View", frame)
        cv2.waitKey(1)

    def shutdown_hook(self):
        rospy.loginfo("Stopping robot...")
        self.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()


if __name__ == '__main__':
    follower = RedBallFollower()
    rospy.on_shutdown(follower.shutdown_hook)
    rospy.spin()
