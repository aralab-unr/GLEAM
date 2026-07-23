from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition   
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('robust_lidar_inertial')

    # Set default arguments
    rviz = LaunchConfiguration('rviz', default='true')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/ouster/points')

    imu_topic = LaunchConfiguration('imu_topic', default='/ouster/imu')
    img_topic = LaunchConfiguration('img_topic', default='/lucid_camera_1/image_raw')

    # Define arguments
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value=rviz,
        description='Launch RViz'
    )
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value=pointcloud_topic,
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value=imu_topic,
        description='IMU topic name'
    )

    declare_img_topic_arg = DeclareLaunchArgument(
        'img_topic',
        default_value=img_topic,
        description='IMAGE topic name'
    )

    # Load parameters
    rls_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'rls.yaml'])
    rls_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])

    # rls Odometry Node
    rls_odom_node = Node(
        package='robust_lidar_inertial',
        executable='rls_odom_node',
        output='screen',
        parameters=[rls_yaml_path, rls_params_yaml_path],
        remappings=[
            ('pointcloud', pointcloud_topic),
            ('imu', imu_topic),
            ('image', img_topic),
            ('odom', 'rls/odom_node/odom'),
            ('pose', 'rls/odom_node/pose'),
            ('path', 'rls/odom_node/path'),
            ('kf_pose', 'rls/odom_node/keyframes'),
            ('kf_cloud', 'rls/odom_node/pointcloud/keyframe'),
            ('deskewed', 'rls/odom_node/pointcloud/deskewed'),
        ],
    )

    # rls Mapping Node
    rls_map_node = Node(
        package='robust_lidar_inertial',
        executable='rls_map_node',
        output='screen',
        parameters=[rls_yaml_path, rls_params_yaml_path],
        remappings=[
            ('keyframes', 'rls/odom_node/pointcloud/keyframe'),
        ],
    )

    # RViz node
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'rls.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rls_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        rls_odom_node,
        rls_map_node,
        rviz_node
    ])
