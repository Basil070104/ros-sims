#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
    PoseStamped
)
from nav_msgs.msg import ( Path, Odometry)
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
import tf
import move_base

import astar
import dijkstras
import math
import orientation

car_orientation = None
car_pose = None

def calculate_trajectory(initial, dest, time):
#   print(initial, dest)

  x,y,z = car_pose
  w = car_orientation[3]

  curr_x, curr_y = float(initial[0]) / 40,(800 - float(initial[1])) / 40
  dest_x, dest_y = float(dest[0]) / 40, (800 - float(dest[1])) / 40

#   print("Received Orienation: x={x}, y={y}, z={z}, w={w}".format(x=x,y=y,z=z,w=w))


  v_x = (dest_x - curr_x) / time
  v_y = (dest_y - curr_y) / time

  print("curr : {x}, {y}; dest : {a}, {b}".format(x=curr_x, y=curr_y, a=dest_x, b=dest_y))

  radians = 0
  if dest_x - curr_x < 0:
    print("left")
    radians = (-1) * math.atan((dest_x - curr_x)/(dest_y - curr_y))
  elif dest_x - curr_x > 0:
    print("right")
    radians = math.atan((dest_x - curr_x)/(dest_y - curr_y))

  v = math.sqrt(v_x**2 + v_y**2) * 0.5

  output = [v, 0, radians]

  return output


def pose_callback(data):
   
    # Pose values
    pose_x = data.pose.position.x
    pose_y = data.pose.position.y
    pose_z = data.pose.position.z

    #Orientation values (Quaternion)
    x = data.pose.orientation.x
    y = data.pose.orientation.y
    z = data.pose.orientation.z
    w = data.pose.orientation.w

    global car_orientation
    car_orientation = [x, y, z, w]

    global car_pose
    car_pose = [pose_x, pose_y, pose_z]

    #print this out 
    # print("Received Orienation: x={x}, y={y}, z={z}, w={w}".format(x=x,y=y,z=z,w=w))

    # return x, y, z, w
  
def run_plan(pub_init_pose, pub_controls, orientation_sub, plan_publish, plan, global_plan_pub):
    init = plan[0]
    send_init_pose(pub_init_pose, init)
    poses_list = list()

    for coor in plan:
        x, y = float(coor[0]) / 40,(800 - float(coor[1])) / 40
        stamped = PoseStamped()
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0

        # pose.orientation.x = 0
        # pose.orientation.y = 0
        # pose.orientation.z = 0
        # pose.orientation.w = 0
        stamped.pose = pose
        poses_list.append(stamped)
    
    header = Header()
    header.frame_id = "/map"
    plan_publish.publish(Path(header=header,poses=poses_list))
    print("Done Printing Path On Rviz . . .")
    goal = poses_list[len(poses_list) - 1]
    time = rospy.get_rostime()
    goal.pose.orientation.z = 0.7
    goal.pose.orientation.w = 0.7
    goal.header.frame_id = "map"
    goal.header.stamp.secs = time.secs
    goal.header.stamp.nsecs = time.nsecs
    goal_pub.publish(goal)

    rate = rospy.Rate(1)
    global_plan_pub.publish(Path(header=header, poses=poses_list))

    while not rospy.is_shutdown():
      global_plan_pub.publish(Path(header=header, poses=poses_list))

      rate.sleep()


    # for i in range(0, len(plan) - 1):
    #     send_command(pub_controls, orientation_sub, plan[i], plan[i + 1])
        
    # for c in plan:
    #     send_command(pub_controls, c)
    
def odom_callback(data):
   
   curr_x = data.pose.pose.position.x
   curr_y = data.pose.pose.position.y

  #  print(curr_x, curr_y)

def send_init_pose(pub_init_pose, init_pose):
    # pose_data = init_pose.split(",")
    # assert len(pose_data) == 3
    pose_data = init_pose

    print(float(pose_data[0]) , float(pose_data[1]))
    x, y, theta = float(pose_data[0]) / 40, (800 - float(pose_data[1])) / 40, math.radians(90)
    print(x , y, theta)
    q = Quaternion(*quaternion_from_euler(0, 0, theta))
    point = Point(x=x, y=y)
    pose = PoseWithCovariance(pose=Pose(position=point, orientation=q))
    pub_init_pose.publish(PoseWithCovarianceStamped(pose=pose))


def send_command(pub_controls, orientation_sub, curr, dest):
    # cmd = c.split(",")
    # assert len(cmd) == 2
    # try:
    #     listener = tf.TransformListener()
    #     (trans, rot)  = listener.lookupTransform( '/car/odom', '/map', rospy.Time(0))
    #     print(trans)
    # except:
    #     print("Listener Failed")

    cmd = calculate_trajectory(curr, dest, 0.05)
    v, delta = float(cmd[0]), float(cmd[2])
    print(v, delta)

    dur = rospy.Duration(secs=1.0)
    rate = rospy.Rate(10)
    start = rospy.Time.now()

    drive = AckermannDrive(steering_angle=delta, speed=v)

    while rospy.Time.now() - start < dur:
        pub_controls.publish(AckermannDriveStamped(drive=drive))
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("path_publisher")

    control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation")
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1)

    init_pose_topic = rospy.get_param("~init_pose_topic", "/initialpose")
    pub_init_pose = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=1)

    orientation_sub = rospy.Subscriber("car/car_pose", PoseStamped, pose_callback)

    plan_publish = rospy.Publisher("car/global_path", Path, queue_size=10)

    odom_sub = rospy.Subscriber("car/vesc/odom", Odometry, odom_callback)

    global_plan_pub = rospy.Publisher("/move_base/TrajectoryPlannerROS/global_plan", Path, queue_size=10)

    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    
    # plan_file = rospy.get_param("~plan_file")

    # with open(plan_file) as f:
    #     plan = f.readlines()
    plan = astar.main()

    # Publishers sometimes need a warm-up time, you can also wait until there
    # are subscribers to start publishing see publisher documentation.
    rospy.sleep(1.0)
    run_plan(pub_init_pose, pub_controls, orientation_sub, plan_publish, plan, global_plan_pub)
    rospy.spin()