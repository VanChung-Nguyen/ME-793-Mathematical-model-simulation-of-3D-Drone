# ME-793-Mathematical-model-simulation-of-3D-Drone

## Software Requirements & Setup

The simulation is configured with the following setup:
- Ubuntu 22.04
- ROS2 Humble
- Gazebo 11
- Xarco-ROS-Humble (sudo apt install ros-humble-xacro)
- Gazebo_ros_pkgs (sudo apt install ros-humble-gazebo-ros-pkgs)
- ACADO Toolkit (https://acado.github.io/)

Follow these commands in order to install the simulation of SM-NMPC for the UAVs on ROS 2:

```shell
# Step 1: Create and build a colcon workspace:
$ mkdir -p ~/ros2_ws/src
$ cd ~/dev_ws/
$ colcon build
$ echo "source ~/ros2_ws/devel/setup.bash" >> ~/.bashrc

# Step 2: Clone this repo into your workspace
$ cd ~/ros2_ws/src
Download the folder smcmpcquad or the smcnmpccube in the main branch

# Step 3: Build the colcon workspace for this package
$ cd ~/ros2_ws
$ colcon build

# Step 4: Export the gazebo models
$ gedit ~/.bashrc
$ Then copy the following line to the bash file: export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/vanchung/ros2_ws/src/smcmpcquad/models

```
* Note that the package contains the code generation and includes the qpOASES library. If the user wants to use SM-NMPC for a different problem, they need to regenerate the code and replace it to the include folder.
* Note that this project uses a custom plugin. Users need to replace the plugin path in the file /urdf/uav_drone.urdf.xacro at line 469. Replace: plugin name="uavplugin" filename="/home/vanchung/ros2_ws/install/smcmpcquad/lib/smcmpcquad/libuavplugin.so" with the correct path by changing the username to the name of your computer. Then rebuild the project again to run the simulation.

## Simulation results

To run the simulation, follow these commands:

```shell
# Step 1: Run the Gazebo model:
$ ros2 launch smcmpcquad model.launch.py

# Step 2: Run the optical flow and controller
$ ros2 run smcmpcquad opticalflownode
$ ros2 run smcmpcquad smcmpcquad

# Step 3: To run the plot 
$ ros2 run smcmpcquad plot
```
