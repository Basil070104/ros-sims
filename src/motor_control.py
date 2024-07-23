#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

def twist_callback(msg):
    ackermann_msg = AckermannDriveStamped()
    ackermann_msg.header.stamp = rospy.Time.now()
    ackermann_msg.drive.speed = msg.linear.x
    ackermann_msg.drive.acceleration = msg.linear.y
    ackermann_msg.drive.steering_angle = msg.angular.x
    ackermann_msg.drive.steering_angle_velocity = msg.angular.z

    ackermann_pub.publish(ackermann_msg)

if __name__ == '__main__':
    rospy.init_node('twist_to_ackermann')

    # Publisher to the ackermann topic
    ackermann_pub = rospy.Publisher('/car/mux/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)

    # Subscriber to cmd_vel
    rospy.Subscriber('/cmd_vel', Twist, twist_callback)

    rospy.spin()