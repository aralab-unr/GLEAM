# robust_lidar_inertial

**RLS-SLAM (Robust LiDAR-Inertial SLAM via Degeneracy-Aware Joint Geometric-Visual Optimization) is a coarse-to-fine framework that uses geometric and visual for joint optimization via challenging scenarios as multi-levels, hybrid, and degeneracy environments.**

# Setup

RLS-SLAM has undergone extensive evaluation across multiple sensor setups and currently supports LiDARs from Ouster, Velodyne, and Hesai. The LiDAR data should be provided as ```sensor_msgs::PointCloud2```, and the IMU measurements should use the format ```sensor_msgs::Imu```.
For optimal performance, the extrinsic calibration among the LiDAR, IMU, and camera sensors relative to the robot’s center of gravity should be defined in the ```cfg/rls.yaml``` configuration file. Even if the visual factor from the camera is not utilized, calibration between the LiDAR and IMU remains necessary. Additionally, strict time synchronization among all sensors is required; otherwise, RLS-SLAM will fail to operate correctly.

<p align='center'>
    <img src="./doc/Flow_chart.png" alt="drawing" width="800"/>
</p>

# Dependencies
The framework has been tested with ROS2 Humble and Ubuntu 22.04. The following configuration, along with the required dependencies, has been verified for compatibility:

- [Ubuntu 22.04](https://releases.ubuntu.com/focal/)
- C++ 14
- [PCL >= 1.10.0](https://pointclouds.org/downloads/)
- [Eigen >= 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [CMake >= 3.12.4]

# Install 
Compile the package using:
```
  mkdir ~/ros2_ws && cd ~/ros2_ws && mkdir src && cd src
```
  Download this anonymous github and copy to the ~/ros2_ws/src
```
  cd ~/ros2_ws
  colcon build
```
# Execution
We evaluate the proposed algorithm against state-of-the-art SLAM methods using the [M2DGR dataset](https://github.com/SJTU-ViSYS/M2DGR), which provides long-term and challenging sequences specifically designed for ground robot applications. To access it quickly, use the following command:
```
  ros2 launch robus_lidar_inertial run_m2dgr.launch.py
```

# Test with your dataset
To test the algorithm with your own dataset, it is essential to specify the camera intrinsic parameters and the LiDAR–camera extrinsic calibration in the ```rls.yaml``` configuration file. If the setup does not include a camera, you must still provide the LiDAR–IMU calibration. Finally, update the sensor topics in ```run.launch.py``` with the correct names and execute the launch file to run the system.

```
  ros2 launch robus_lidar_inertial run.launch.py
```

# Demo
In accordance with the double-blind review policy, the dataset will be released after a decision is made on this submission. Below are sample results presented in the paper. The full demonstration video with comparative analysis can be viewed in the accompanying video.
<p align='center'>
    <img src="./doc/stair-case.gif" alt="drawing" width="400"/>
    <img src="./doc/hybrid-scale.gif" alt="drawing" width="400"/>
    <img src="./doc/culvert.gif" alt="drawing" width="600"/>
</p>
