# MuSHR Autonomous Navigation Project

## Overview
This project implements autonomous navigation for the Multi-agent System for non-Holonomic Racing (MuSHR) car using the Robotic Operating System (ROS). The system integrates environmental data processing, mapping, path-planning algorithms, and trajectory generation to enable autonomous driving experiments.

## Features
- **Mapping & Localization:** Converts environmental data into a usable map.
- **Path-Planning Algorithms:** Implements three separate algorithms for efficient navigation.
- **Cubic Spline Trajectory Generation:** Ensures smooth and optimal movement.
- **ROS Integration:** Uses various ROS nodes for communication and control.

## Dependencies
Ensure the following packages and dependencies are installed:

- ROS (Noetic or Melodic)
- MuSHR stack
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- `ackermann_msgs` for vehicle control

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/Basil070104/main.git
   cd mushr-navigation
   ```
2. Install dependencies:
   ```sh
   sudo apt-get update && sudo apt-get install -y ros-noetic-ackermann-msgs python3-numpy python3-opencv python3-matplotlib
   ```
3. Build the ROS workspace:
   ```sh
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage
1. Launch the MuSHR environment:
   ```sh
   roslaunch mushr_navigation simulation.launch
   ```
2. Run the path-planning node:
   ```sh
   rosrun mushr_navigation path_planner.py
   ```
3. Monitor the vehicle movement using RViz:
   ```sh
   roslaunch mushr_navigation visualization.launch
   ```

## ROS Nodes
### `/car/mux/ackermann_cmd_mux`
- Muxes control inputs and forwards commands to the vehicle.
- Accepts `AckermannDriveStamped` messages.

### `/car/vesc/ackermann_to_vesc`
- Converts `AckermannDriveStamped` messages to motor commands.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Contact
For inquiries or contributions, contact Basil Khwaja at [your email].

