#!/usr/bin/env python3

import rospy
import actionlib
from math import radians
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler


class MultiGoalNavigator:
    def __init__(self):
        rospy.init_node('multi_goal_navigator')

        rospy.loginfo("Waiting for move_base action server...")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base.")

    def create_goal(self, x, y, yaw_deg):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        yaw_rad = radians(yaw_deg)
        q = quaternion_from_euler(0, 0, yaw_rad)

        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        return goal

    def send_goal(self, goal, goal_name):
        rospy.loginfo(f"Sending goal: {goal_name}")
        self.client.send_goal(goal)
        self.client.wait_for_result()

        state = self.client.get_state()

        if state == 3:  # GoalStatus.SUCCEEDED
            rospy.loginfo(f"Reached {goal_name}")
            return True
        else:
            rospy.logwarn(f"Failed to reach {goal_name}, state = {state}")
            return False

    def run(self):
        # Replace these with your actual map coordinates from the lab
        L1 = (0.0, 0.0, 0.0)
        L2 = (2.0, 0.0, 90.0)
        L3 = (2.0, 2.0, 180.0)

        goals = [
            ("L2", self.create_goal(*L2)),
            ("L3", self.create_goal(*L3)),
            ("L1", self.create_goal(*L1)),
        ]

        for name, goal in goals:
            success = self.send_goal(goal, name)

            if not success:
                rospy.logwarn("Stopping sequence because one goal failed.")
                break

            rospy.sleep(2)

        rospy.loginfo("Navigation sequence finished.")


if __name__ == "__main__":
    try:
        navigator = MultiGoalNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
