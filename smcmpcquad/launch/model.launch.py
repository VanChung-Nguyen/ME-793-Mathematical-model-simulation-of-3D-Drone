import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Path to the Gazebo launch file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': os.path.join(
                get_package_share_directory('smcmpcquad'),
                'worlds',
                'ground.world'
            )
        }.items()
    )

    # Get the UAV URDF file
    package_share_directory = get_package_share_directory('smcmpcquad')
    urdf_file_path = os.path.join(package_share_directory, 'urdf', 'uav_drone.urdf.xacro')

    doc = xacro.process_file(urdf_file_path)
    urdf_xml = doc.toprettyxml(indent='  ')

    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml}]
    )

    # Spawn UAV in Gazebo
    spawn_entity_node = Node(
    package='gazebo_ros',
    executable='spawn_entity.py',
    name='spawn_entity',
    output='screen',
    arguments=[
        '-topic', 'robot_description',
        '-entity', 'quad',
        '-x', '0', '-y', '0', '-z', '0.2',
    ]
    )
    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher_node,
        spawn_entity_node
    ])
