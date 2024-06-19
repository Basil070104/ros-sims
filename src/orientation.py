import rospy
from geometry_msgs.msg import PoseStamped


def pose_callback(data):
   
  x = data.orientation.x
  y = data.orientation.y
  z = data.orientation.z
  w = data.orientation.w

  #print this out 
  print("Received Orienation: x={x}, y={y}, z={z}, w={w}".format(x,y,z,w))

def pose_listener():
  rospy.init_node("orientation_listener", anonymous=True)
  rospy.Subscriber("/car/car_pese", PoseStamped, pose_callback)
  rospy.spin()


def main():
  pose_listener()

if __name__ == "__main__":
  main()