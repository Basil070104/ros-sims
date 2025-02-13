#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
    PoseStamped,
    Twist
)
from nav_msgs.msg import ( Path, Odometry)
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler
import tf
import move_base
import numpy as np
import matplotlib.pyplot as plt

import astar
import dijkstras
import math
import orientation



class Simulation:
  
  def __init__(self, method):
    car_orientation = None
    car_pose = None

    curr_x, curr_y, quat_w = 0, 0, 0
    self.x_cubic = None
    self.y_cubic = None
    self.goal_pub = None
    self.cmd_vel_pub = None
    self.t_interp = None
    self.method = method

  def calculate_trajectory(self, initial, dest, time):

    x,y,z = car_pose
    w = car_orientation[3]

    curr_x, curr_y = float(initial[0]) / 40,(800 - float(initial[1])) / 40
    dest_x, dest_y = float(dest[0]) / 40, (800 - float(dest[1])) / 40
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

  def pose_callback(self, data):
    
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

  def calc_vel(self, d_x, d_y, t):
    return math.sqrt(d_x(t)**2 + d_y(t)**2)

  def calc_ang(self, d_x, d_y, dd_x, dd_y, t):
    curve = ((d_x(t) * dd_y(t)) - (d_y(t) * dd_x(t)))/(math.pow(d_x(t)**2 + d_y(t)**2, 3/2))
    return curve * self.calc_vel(d_x, d_y, t)

  def heading(self, dx, dy, t):
    return math.atan2(dy(t), dx(t))

  def calc_accel(self, ddx, ddy, t):
    return math.sqrt((ddx(t) * ddx(t)) + (ddy(t) * ddy(t)))

  def compute_twist(self,current_pose, target_pose):
      twist = Twist()

      # Compute the difference in position
      dx = target_pose.pose.position.x - current_pose.pose.position.x
      dy = target_pose.pose.position.y - current_pose.pose.position.y

      # Compute the distance to the target
      distance = (dx**2 + dy**2)**0.5

      # Compute the angle to the target
      target_yaw = tf.transformations.euler_from_quaternion([
          target_pose.pose.orientation.x,
          target_pose.pose.orientation.y,
          target_pose.pose.orientation.z,
          target_pose.pose.orientation.w
      ])[2]

      current_yaw = tf.transformations.euler_from_quaternion([
          current_pose.pose.orientation.x,
          current_pose.pose.orientation.y,
          current_pose.pose.orientation.z,
          current_pose.pose.orientation.w
      ])[2]

      angle_to_target = target_yaw - current_yaw
      twist.linear.x = distance 
      twist.angular.z = angle_to_target

      return twist

  def plan(self,pub_init_pose, pub_controls, orientation_sub, plan_publish, plan, global_plan_pub):
      init = self.x_cubic(0), self.y_cubic(0)
      print(init)
      # raise SystemExit
      self.send_init_pose(pub_init_pose, init)
      poses_list = list()
      time = rospy.get_rostime()

      for coor in range(0, len(plan) - 1):
          coordinate = plan[coor]
          x, y = float(coordinate[0]) / 40,(800 - float(coordinate[1])) / 40
          stamped = PoseStamped()
          pose = Pose()
          pose.position.x = x
          pose.position.y = y
          pose.position.z = 0

          future_coordinate = plan[coor + 1]
          future_x , future_y = float(future_coordinate[0]) / 40,(800 - float(future_coordinate[1])) / 40

          angle = math.atan2(future_y - y, future_x - x)

          pose.orientation.z = 0.7
          pose.orientation.w = angle
          stamped.pose = pose
          stamped.header.frame_id= "map"
          stamped.header.stamp.secs = time.secs
          stamped.header.stamp.nsecs = time.nsecs
          poses_list.append(stamped)
      
      header = Header()
      header.frame_id = "/map"
      plan_publish.publish(Path(header=header,poses=poses_list))
      print("Done Printing Path On Rviz . . .")
      goal = poses_list[len(poses_list) - 1]
      goal.pose.orientation.z = 0.7
      goal.pose.orientation.w = 0.7
      goal.header.frame_id = "map"
      goal.header.stamp.secs = time.secs
      goal.header.stamp.nsecs = time.nsecs
      self.goal_pub.publish(goal)

      rate = rospy.Rate(1)
      global_plan_pub.publish(Path(header=header, poses=poses_list))

      rate = rospy.Rate(10)
      index = 1
      fig , ax = plt.subplots(1,3)
      fig.set_size_inches(15,8)
      fig.tight_layout()
      while index < len(self.t_interp):
        t = self.t_interp[index]
        val = self.calc_vel(self.x_cubic.derivative(nu=1), self.y_cubic.derivative(nu=1), t)
        ang = self.calc_ang(self.x_cubic.derivative(nu=1), self.y_cubic.derivative(nu=1), self.x_cubic.derivative(nu=2), self.y_cubic.derivative(nu=2), t)
        head = self.heading(self.x_cubic.derivative(nu=1), self.y_cubic.derivative(nu=1), t)
        accel = self.calc_accel(self.x_cubic.derivative(nu=2), self.y_cubic.derivative(nu=2), t)
        twist = Twist()
        print(t, self.x_cubic(t), self.y_cubic(t), curr_x, curr_y)
        # ax[0].plot((spline_x(t) / 40), ((800 - spline_y(t)) / 40), "o", color="purple")
        ax[0].plot((self.x_cubic(t)), ((self.y_cubic(t))), "o-", color="purple", label="Desired")
        ax[0].plot( curr_x, curr_y, "o-", color="red", label="Actual")
        ax[0].legend(loc="lower left")
        ax[0].set_title("Desired vs. Actual")
        ax[1].plot(t, val, 'o-', color="blue")
        ax[1].set_title("Speed")
        ax[2].plot(t, ang * 5, 'o-', color="green")
        ax[2].set_title("Angular Velocity")
        twist.linear.x = val * 1.03
        # twist.linear.y = accel / 40
        twist.angular.x = ang * 4.75
        # twist.angular.z = ang 
        self.cmd_vel_pub.publish(twist)
        index+=1
        rate.sleep()
      
      plt.show()
      
  def odom_callback(self,data):
    
    global curr_x, curr_y, quat_w

    curr_x = data.pose.position.x
    curr_y = data.pose.position.y

    quat_w = data.pose.orientation.w

    return curr_x, curr_y, quat_w

  def send_init_pose(self,pub_init_pose, init_pose):
      pose_data = init_pose

      print(float(pose_data[0]) , float(pose_data[1]))
      x, y, theta = pose_data[0], pose_data[1], self.heading(self.x_cubic.derivative(nu=1), self.y_cubic.derivative(1), 0.1)
      print(x , y, self.heading(self.x_cubic.derivative(nu=1), self.y_cubic.derivative(1), 0.1))
      q = Quaternion(*quaternion_from_euler(0, 0, theta))
      point = Point(x=x, y=y)
      pose = PoseWithCovariance(pose=Pose(position=point, orientation=q))
      pub_init_pose.publish(PoseWithCovarianceStamped(pose=pose))

  def send_command(self, pub_controls, orientation_sub, curr, dest):

      cmd = self.calculate_trajectory(curr, dest, 0.05)
      v, delta = float(cmd[0]), float(cmd[2])
      print(v, delta)

      dur = rospy.Duration(secs=1.0)
      rate = rospy.Rate(10)
      start = rospy.Time.now()

      drive = AckermannDrive(steering_angle=delta, speed=v)

      while rospy.Time.now() - start < dur:
          pub_controls.publish(AckermannDriveStamped(drive=drive))
          rate.sleep()

  def run(self):
    rospy.init_node("path_publisher")

    control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation")
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1)

    init_pose_topic = rospy.get_param("~init_pose_topic", "/initialpose")
    pub_init_pose = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=1)

    orientation_sub = rospy.Subscriber("car/car_pose", PoseStamped, self.pose_callback)

    plan_publish = rospy.Publisher("car/global_path", Path, queue_size=10)

    odom_sub = rospy.Subscriber("car/car_pose", PoseStamped, self.odom_callback)

    global_plan_pub = rospy.Publisher("/move_base/TrajectoryPlannerROS/global_plan", Path, queue_size=10)

    self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

    self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    spline_pub = rospy.Publisher("car/spline", Path, queue_size=1)
    
    plan, self.x_cubic, self.y_cubic, spline_x, spline_y, t_arr = self.method.main()

    spline_list = list()
    time = rospy.get_rostime()
    self.t_interp = np.linspace(np.min(t_arr), np.max(t_arr), np.max(t_arr) / 0.1)

    for t in self.t_interp: 
      x, y = self.x_cubic(t), self.y_cubic(t)
      stamped = PoseStamped()
      pose = Pose()
      pose.position.x = x
      pose.position.y = y
      pose.position.z = 0

      pose.orientation.z = 0.7
      pose.orientation.w = 0.7

      stamped.pose = pose
      stamped.header.frame_id= "map"
      stamped.header.stamp.secs = time.secs
      stamped.header.stamp.nsecs = time.nsecs
      spline_list.append(stamped)
       

    header = Header()
    header.frame_id = "/map"
    spline_pub.publish(Path(header=header,poses=spline_list))

    # Publishers sometimes need a warm-up time, you can also wait until there
    # are subscribers to start publishing see publisher documentation.
    rospy.sleep(1.0)
    self.plan(pub_init_pose, pub_controls, orientation_sub, plan_publish, plan, global_plan_pub)
    rospy.spin()
    
if __name__ == "__main__":
    sim = Simulation(method=astar)
    sim.run()