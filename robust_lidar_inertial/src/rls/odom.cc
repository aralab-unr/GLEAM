#include "rls/odom.h"
#include "rls/utils.h"

#include <queue>

#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include <fstream>
#include <iomanip>
#include <cstdlib> // For system()
#include <pcl/io/ply_io.h>

rls::OdomNode::OdomNode() : Node("rls_odom_node") {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->rls_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  // Initialization
  this->rgb_colorized = false;
  this->visual_initialization = false;
  this->valid_point_flag      = false;
  this->mahalanobis_idx_      = false;

  // LiDAR Subscriber
  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud", 1,
      std::bind(&rls::OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);
  
  // IMU Subcriber
  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::SensorDataQoS(),
      std::bind(&rls::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);
  
  // Image Subcriber
  this->img_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto img_sub_opt   = rclcpp::SubscriptionOptions();
  img_sub_opt.callback_group = this->img_cb_group;
  this->img_sub      = this->create_subscription<sensor_msgs::msg::Image>("image", rclcpp::SensorDataQoS(),
      std::bind(&rls::OdomNode::callbackImage, this, std::placeholders::_1), img_sub_opt);
  
  // Publisher
  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 1);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);

  this->bev_color_pub_ = this->create_publisher<sensor_msgs::msg::Image>("bev_colored", 10);

  // Intensity cylindrical image
  this->intensity_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
    "lidar/intensity_image", 10);

  // For debug only
  this->debug_prior_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("debug/pose_prior", 1);
  this->debug_final_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("debug/pose_final", 1);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01), 
      std::bind(&rls::OdomNode::publishPose, this));
  
  // Additional Publisher
  this->pubImage = image_transport::create_publisher(this, "rgb_img");

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // Modification
  this->last_timestamp_img = 0.0;
  this->lid_num            = 0.0;
  this->first_write        = true;

  this->camera.intrinsic_matrix << this->camera.fx, 0, this->camera.cx,
                                    0, this->camera.fy, this->camera.cy,
                                    0, 0, 1;

  // Camera instrinsic params


  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

  // =================================================================
  // [NEW] Load Vibration Mask for Vibration-Aware SLAM
  // =================================================================
  this->vibration_mask_global_ = std::make_shared<pcl::PointCloud<PointType>>();
  std::string mask_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/all_vibration_rois.pcd";
  
  if (pcl::io::loadPCDFile<PointType>(mask_path, *this->vibration_mask_global_) == -1) {
      RCLCPP_WARN(this->get_logger(), "No vibration mask found at %s. Running Standard SLAM.", mask_path.c_str());
      this->use_vibration_mask_ = false;
  } else {
      RCLCPP_INFO(this->get_logger(), "Loaded Vibration Mask (%zu points). Enabling Vibration-Aware SLAM!", this->vibration_mask_global_->size());
      this->vibration_mask_kdtree_ = std::make_shared<pcl::KdTreeFLANN<PointType>>();
      this->vibration_mask_kdtree_->setInputCloud(this->vibration_mask_global_);
      this->use_vibration_mask_ = false;
  }

}

// rls::OdomNode::~OdomNode() {}

rls::OdomNode::~OdomNode() {
  // 1. Existing code: Safely close the JSON array so Nerfstudio doesn't crash
    std::string json_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data/transforms.json";
    std::ofstream json_file(json_path, std::ios_base::app);
    if (json_file.is_open()) {
        json_file << "\n  ]\n}\n";
        json_file.close();
        RCLCPP_INFO(this->get_logger(), "Successfully closed transforms.json for Splatting.");
    }

    // 2. NEW CODE: Stitch and Save the Global Point Cloud
    RCLCPP_INFO(this->get_logger(), "Stitching and saving Global Point Cloud... Please wait.");
    pcl::PointCloud<PointType>::Ptr global_map = std::make_shared<pcl::PointCloud<PointType>>();

    // Lock the keyframes and stitch them together
    // (Note: Your buildKeyframesAndSubmap function already transforms these keyframes into the global frame)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    for (size_t i = 0; i < this->keyframes.size(); i++) {
        if (this->keyframes[i].second != nullptr && !this->keyframes[i].second->empty()) {
            *global_map += *this->keyframes[i].second;
        }
    }
    lock.unlock();

    if (!global_map->empty()) {
        // Optional: Voxel filter the final map to reduce file size and clean up density
        pcl::PointCloud<PointType>::Ptr filtered_map = std::make_shared<pcl::PointCloud<PointType>>();
        pcl::VoxelGrid<PointType> ds_filter;
        ds_filter.setLeafSize(0.01f, 0.01f, 0.01f); // 5cm resolution
        ds_filter.setInputCloud(global_map);
        ds_filter.filter(*filtered_map);

        // Save to PCD
        std::string map_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/global_map.pcd";
        pcl::io::savePCDFileBinary(map_path, *filtered_map);
        RCLCPP_INFO(this->get_logger(), "Successfully saved global map (%zu points) to: %s", filtered_map->size(), map_path.c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "Global map was empty, nothing to save.");
    }

    std::string time_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/cpu_times_pruned.csv";
    std::ofstream time_file(time_path);
    if (time_file.is_open()) {
        time_file << "Frame,Time_ms\n";
        for (size_t i = 0; i < this->comp_times.size(); i++) {
            time_file << i << "," << (this->comp_times[i] * 1000.0) << "\n";
        }
        time_file.close();
        RCLCPP_INFO(this->get_logger(), "Saved CPU Latency data to %s", time_path.c_str());
    }
}

void rls::OdomNode::getParams() {

  // Version
  rls::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  rls::declare_param(this, "frames/odom", this->odom_frame, "odom");
  rls::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  rls::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  rls::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  rls::declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  rls::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  rls::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  rls::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  rls::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  rls::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  rls::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  rls::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  rls::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  rls::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  rls::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  rls::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  rls::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  rls::declare_param(this, "adaptive", this->adaptive_params_, true);

  // Alignment Parameters
  rls::declare_param(this, "alignment", this->align_paramas_, true);
  rls::declare_param(this, "map3d", this->map3d_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  rls::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  rls::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  rls::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  rls::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  rls::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  rls::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  rls::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  rls::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  rls::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  rls::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  rls::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  rls::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // Camera specs
  this->camera.width  = this->declare_parameter<int>("params/cam/width", 1280);
  this->camera.height = this->declare_parameter<int>("params/cam/height", 720);
  this->camera.fx     = this->declare_parameter<double>("params/cam/fx", 1009.32421875);
  this->camera.fy     = this->declare_parameter<double>("params/cam/fy", 1009.32421875);
  this->camera.cx     = this->declare_parameter<double>("params/cam/cx", 632.487060546875);
  this->camera.cy     = this->declare_parameter<double>("params/cam/cy", 358.5774841308594);

  this->detector_ = cv::FastFeatureDetector::create(20, true);
  // [NEW] Debug Publishers
  this->debug_phase_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug/phase_image", 1);
  this->debug_phase_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("debug/phase_pose", 1);

  RCLCPP_INFO(this->get_logger(), "Camera parameters loaded: %dx%d  fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
              camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy);

  std::vector<double> cam2lidar_t;
  rls::declare_param(this, "extrinsics/cam2lidar/t", cam2lidar_t, t_default);

  std::vector<double> cam2lidar_R;
  rls::declare_param(this, "extrinsics/cam2lidar/R", cam2lidar_R, R_default);

  Eigen::Matrix3d R_temp = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(cam2lidar_R.data());
  this->Rcl = R_temp.cast<float>();
  Eigen::Vector3d temp = Eigen::Vector3d(cam2lidar_t[0], cam2lidar_t[1], cam2lidar_t[2]);
  this->Pcl = temp.cast<float>().transpose();

  std::cout << "T_cam2lidar:\n" << this->Pcl << std::endl;
  std::cout << "R_cam2lidar:\n" << this->Rcl << std::endl;

  // std::cout << "T_baselink2lidar:\n" << this->extrinsics.baselink2lidar.t << std::endl;
  // std::cout << "R_baselink2lidar :\n" << this->extrinsics.baselink2lidar.R << std::endl;
 
  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  rls::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  rls::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  rls::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  rls::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  rls::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  rls::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  rls::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  rls::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  rls::declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0);
  rls::declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0);
  rls::declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0);
  rls::declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0);
  rls::declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0);
  rls::declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0);
  rls::declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0);

  // [NEW] LiDAR Resolution Parameters for Visual Projection
  // Defaults set for Ouster OS1-64 (Gen 2/3)
  rls::declare_param(this, "lidar/lines", this->scan_H_, 64); 
  rls::declare_param(this, "lidar/cols", this->scan_W_, 1024); 
  rls::declare_param(this, "lidar/fov_up", this->scan_fov_up_, 22.5); 
  rls::declare_param(this, "lidar/fov_down", this->scan_fov_down_, -22.5);

  // Velodyne VLP-32C FOV M2DGR dataset: 
  // rls::declare_param(this, "lidar/lines", this->scan_H_, 32); 
  // rls::declare_param(this, "lidar/cols", this->scan_W_, 1800); 
  // rls::declare_param(this, "lidar/fov_up", this->scan_fov_up_,15.0); 
  // rls::declare_param(this, "lidar/fov_down", this->scan_fov_down_, -25.0);
  
  // Pre-compute vertical resolution
  this->fov_vertical_rad_ = (std::abs(this->scan_fov_up_) + std::abs(this->scan_fov_down_)) * M_PI / 180.0;
  this->fov_down_rad_ = this->scan_fov_down_ * M_PI / 180.0;
}

void rls::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|                     Robust LiDAR-Inertial SLAM                    |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void rls::OdomNode::publishPose() {

  // nav_msgs::msg::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub->publish(this->pose_ros);

}

void rls::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::msg::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub->publish(this->path_ros);

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br->sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br->sendTransform(transformStamped);

}

void rls::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  if (!this->map3d_params_) {
    sensor_msgs::msg::PointCloud2 deskewed_ros;
    pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
    deskewed_ros.header.stamp = this->scan_header_stamp;
    deskewed_ros.header.frame_id = this->odom_frame;
    this->deskewed_pub->publish(deskewed_ros);
  }
}

void rls::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
  }

}

// void rls::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

//   // 1. Convert ROS msg to PCL (This preserves the 1024x64 grid structure)
//   pcl::PointCloud<PointType>::Ptr temp_scan = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::fromROSMsg(*pc, *temp_scan);

//   // [CRITICAL FIX] Save the Organized Cloud for Vision
//   // We need the NaNs and the grid structure so (u,v) lookup works.
//   this->full_organized_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*temp_scan);

//   // 2. Generate Image (Using the Organized Cloud)
//   this->current_intensity_img = this->generateIntensityImage(this->full_organized_scan_);

//   // 3. NOW we perform filtering for GICP (Destroys structure, but fine for GICP)
//   pcl::PointCloud<PointType>::Ptr processed_scan = std::make_shared<pcl::PointCloud<PointType>>();
  
//   // Remove NaNs (Destroys organization)
//   std::vector<int> idx;
//   pcl::removeNaNFromPointCloud(*temp_scan, *processed_scan, idx);

//   // Crop Box Filter
//   this->crop.setInputCloud(processed_scan);
//   this->crop.filter(*processed_scan);

//   // Sensor detection logic (Keep your existing logic)
//   this->sensor = rls::SensorType::UNKNOWN;
//   for (auto &field : pc->fields) {
//     if (field.name == "t") { this->sensor = rls::SensorType::OUSTER; break; }
//     else if (field.name == "time") { this->sensor = rls::SensorType::VELODYNE; break; }
//     else if (field.name == "timestamp" && temp_scan->points[0].timestamp < 1e14) { this->sensor = rls::SensorType::HESAI; break; }
//     else if (field.name == "timestamp" && temp_scan->points[0].timestamp > 1e14) { this->sensor = rls::SensorType::LIVOX; break; }
//   }

//   if (this->sensor == rls::SensorType::UNKNOWN) {
//     this->deskew_ = false;
//   }

//   this->scan_header_stamp = pc->header.stamp;
  
//   // 4. Save the processed (filtered) cloud for GICP
//   this->original_scan = processed_scan; 
// }

void rls::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  // 1. Convert ROS msg to PCL
  pcl::PointCloud<PointType>::Ptr temp_scan = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *temp_scan);

  // -----------------------------------------------------------------------
  // [CRITICAL FIX] Handle Unorganized Clouds (M2DGR / Velodyne)
  // -----------------------------------------------------------------------
  if (temp_scan->height == 1) {
      
      // We must organize this flat cloud into a grid matching our YAML config
      int H = this->scan_H_; // e.g., 32
      int W = this->scan_W_; // e.g., 1800
      
      pcl::PointCloud<PointType>::Ptr organized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
      organized_cloud->height = H;
      organized_cloud->width = W;        // [FIXED TYPO]
      organized_cloud->is_dense = false; // [FIXED TYPO]
      organized_cloud->points.resize(H * W);

      // Initialize with NaNs
      PointType nan_point;
      nan_point.x = std::numeric_limits<float>::quiet_NaN();
      nan_point.y = std::numeric_limits<float>::quiet_NaN();
      nan_point.z = std::numeric_limits<float>::quiet_NaN();
      std::fill(organized_cloud->points.begin(), organized_cloud->points.end(), nan_point);

      // Pre-compute angles
      float fov_up_rad = (this->scan_fov_up_ * M_PI) / 180.0;
      float fov_down_rad = (this->scan_fov_down_ * M_PI) / 180.0;
      float fov_span = std::abs(fov_up_rad) + std::abs(fov_down_rad);

      // Project points into the grid
      for (const auto& p : temp_scan->points) {
          if (!std::isfinite(p.x)) continue;
          
          float range = std::hypot(p.x, p.y, p.z);
          if (range < 0.5) continue;

          // Spherical Projection
          float yaw = -std::atan2(p.y, p.x);
          float pitch = std::asin(p.z / range);

          // Map to Image Coordinates (u, v)
          float proj_x = 0.5 * (yaw / M_PI + 1.0) * W;
          float proj_y = 1.0 - (pitch + std::abs(fov_down_rad)) / fov_span;
          proj_y *= H;

          int u = std::clamp(static_cast<int>(proj_x), 0, W - 1);
          int v = std::clamp(static_cast<int>(proj_y), 0, H - 1);

          // Place point in the Organized Cloud
          organized_cloud->at(u, v) = p; 
      }

      this->full_organized_scan_ = organized_cloud;
  
  } 
  else {
      // Ouster / Hesai (Already Organized)
      this->full_organized_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*temp_scan);
  }

  // -----------------------------------------------------------------------

  // 2. Generate Image (Now safe because full_organized_scan_ is guaranteed to be a grid)
  this->current_intensity_img = this->generateIntensityImage(this->full_organized_scan_);

  // 3. Filtering for GICP (Standard)
  pcl::PointCloud<PointType>::Ptr processed_scan = std::make_shared<pcl::PointCloud<PointType>>();
  std::vector<int> idx;
  pcl::removeNaNFromPointCloud(*temp_scan, *processed_scan, idx); // Use original points for GICP

  this->crop.setInputCloud(processed_scan);
  this->crop.filter(*processed_scan);

  // Sensor Type Detection
  this->sensor = rls::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") { this->sensor = rls::SensorType::OUSTER; break; }
    else if (field.name == "time") { this->sensor = rls::SensorType::VELODYNE; break; }
    else if (field.name == "timestamp" && temp_scan->points[0].timestamp < 1e14) { this->sensor = rls::SensorType::HESAI; break; }
    else if (field.name == "timestamp" && temp_scan->points[0].timestamp > 1e14) { this->sensor = rls::SensorType::LIVOX; break; }
  }

  if (this->sensor == rls::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = processed_scan; 
}

void rls::OdomNode::preprocessPoints() {

  // Deskew the original rls-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
    frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void rls::OdomNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>(1, this->original_scan->points.size());
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == rls::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == rls::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == rls::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };
  } else if (this->sensor == rls::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;

  Eigen::Vector3f prior_trans = this->lidarPose.p;
  this->timestampsTest = timestamps;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    RCLCPP_FATAL(this->get_logger(),"Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void rls::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

}

// void rls::OdomNode::setInputSource() {
//   this->gicp.setInputSource(this->current_scan);
//   this->gicp.calculateSourceCovariances();
// }

// void rls::OdomNode::setInputSource() {
  
//   // =================================================================
//   // THE DEGENERACY STRESS TEST (For Paper Proof)
//   // We crop out the "safe" background walls so the solver is FORCED
//   // to rely on the vibrating structure. 
//   // =================================================================
//   pcl::PointCloud<PointType>::Ptr stressed_scan = std::make_shared<pcl::PointCloud<PointType>>();
  
//   // We only keep points within 4.0 meters of the robot. 
//   // This simulates a severe FOV restriction / heavy fog / degenerate tunnel.
//   for (const auto& pt : this->current_scan->points) {
//       if (std::isfinite(pt.x) && (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z) < (4.0 * 4.0)) {
//           stressed_scan->push_back(pt);
//       }
//   }
//   this->current_scan = stressed_scan; // Overwrite scan with blinded version

//   // 1. Let GICP calculate covariances on the blinded scan
//   this->gicp.setInputSource(this->current_scan);
//   this->gicp.calculateSourceCovariances();

//   // =================================================================
//   // [THEORETICAL CONTRIBUTION] Spatiotemporal Covariance Decomposition
//   // =================================================================
//   if (this->use_vibration_mask_ && this->vibration_mask_kdtree_ && this->vibration_mask_global_->size() > 0) {
      
//       auto const_source_covs = this->gicp.getSourceCovariances();
      
//       if (!const_source_covs || const_source_covs->size() != this->current_scan->points.size()) {
//           return;
//       }
      
//       auto mutable_source_covs = std::make_shared<nano_gicp::CovarianceList>(*const_source_covs);
      
//       int relaxed_count = 0;
//       float tau_sq = 1.0f * 1.0f; // 1.0m search radius
//       double vibration_penalty = 0.5; 

//       Eigen::Matrix4f T_local_to_global = this->T_prior * this->extrinsics.baselink2lidar_T;

//       for (size_t i = 0; i < this->current_scan->points.size(); ++i) {
//           const auto& pt = this->current_scan->points[i];
          
//           Eigen::Vector4f pt_global = T_local_to_global * pt.getVector4fMap();
//           PointType p_g;
//           p_g.x = pt_global(0); p_g.y = pt_global(1); p_g.z = pt_global(2);

//           std::vector<int> indices(1);
//           std::vector<float> sq_dists(1);

//           if (this->vibration_mask_kdtree_->nearestKSearch(p_g, 1, indices, sq_dists) > 0) {
//               if (sq_dists[0] <= tau_sq) {
                  
//                   Eigen::Matrix4d cov_4d = (*mutable_source_covs)[i];
//                   Eigen::Matrix3d cov_3d = cov_4d.block<3,3>(0,0);
                  
//                   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(cov_3d);
//                   if (eigen_solver.info() == Eigen::Success) {
//                       Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues(); 
//                       Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();
                      
//                       // Inject SCD-ICP theoretical penalty
//                       eigenvalues(0) += vibration_penalty;
                      
//                       Eigen::Matrix3d new_cov_3d = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
//                       cov_4d.block<3,3>(0,0) = new_cov_3d;
//                       (*mutable_source_covs)[i] = cov_4d; 
                      
//                       relaxed_count++;
//                   }
//               }
//           }
//       }
//       this->gicp.setSourceCovariances(mutable_source_covs);
//       RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
//          "SCD-ICP: Applied normal-vector covariance inflation to " << relaxed_count << " points.");
//   }
// }

// void rls::OdomNode::setInputSource() {
  
//   // 1. Give the ENTIRE cloud to GICP first. Do NOT delete points.
//   this->gicp.setInputSource(this->current_scan);
//   this->gicp.calculateSourceCovariances();

//   // =================================================================
//   // [NEW] VIBRATION-AWARE COVARIANCE RELAXATION (Soft Masking)
//   // Instead of deleting points, we inject high uncertainty into 
//   // the vibrating regions so GICP treats them as "Fuzzy Anchors".
//   // =================================================================
//   if (this->use_vibration_mask_ && this->vibration_mask_kdtree_) {
      
//       // Grab the READ-ONLY covariances that GICP just calculated
//       auto const_source_covs = this->gicp.getSourceCovariances();
      
//       // Create a MUTABLE COPY of the covariances so we can modify them
//       auto mutable_source_covs = std::make_shared<nano_gicp::CovarianceList>(*const_source_covs);
      
//       int relaxed_count = 0;
      
//       // KD-Tree Search Radius (1.0 meters to catch ghosts)
//       float tau_sq = 1.0f * 1.0f; 

//       // Transform local scan to the global frame to query the mask
//       Eigen::Matrix4f T_local_to_global = this->T_prior * this->extrinsics.baselink2lidar_T;

//       for (size_t i = 0; i < this->current_scan->points.size(); ++i) {
//           const auto& pt = this->current_scan->points[i];
//           if (!std::isfinite(pt.x)) continue;

//           // Project point to global frame
//           Eigen::Vector4f pt_global = T_local_to_global * pt.getVector4fMap();
//           PointType p_g;
//           p_g.x = pt_global(0); p_g.y = pt_global(1); p_g.z = pt_global(2);

//           // Query the Vibration KDTree
//           std::vector<int> indices(1);
//           std::vector<float> sq_dists(1);

//           if (this->vibration_mask_kdtree_->nearestKSearch(p_g, 1, indices, sq_dists) > 0) {
//               if (sq_dists[0] <= tau_sq) {
                  
//                   // 🚨 THE MAGIC: COVARIANCE INFLATION 🚨
//                   Eigen::Matrix4d fuzzy_cov = Eigen::Matrix4d::Identity();
//                   fuzzy_cov(0,0) = 2.0; // 2 meters of uncertainty in X
//                   fuzzy_cov(1,1) = 2.0; // 2 meters of uncertainty in Y
//                   fuzzy_cov(2,2) = 2.0; // 2 meters of uncertainty in Z
//                   fuzzy_cov(3,3) = 1e-6; // Homogeneous scale
                  
//                   // Assign the inflated matrix to our MUTABLE copy
//                   (*mutable_source_covs)[i] = fuzzy_cov; 
//                   relaxed_count++;
//               }
//           }
//       }
      
//       // Give the heavily modified covariances BACK to GICP to use for optimization
//       this->gicp.setSourceCovariances(mutable_source_covs);
      
//       RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
//          "Vibration-Aware ICP: Applied Covariance Relaxation to " << relaxed_count << " unstable points.");
//   }
// }

void rls::OdomNode::setInputSource() {
  
  // =================================================================
  // COMPUTATIONAL PRUNING (Lazy ICP)
  // Drop vibrating points BEFORE we waste CPU cycles calculating 
  // their surface normals and covariance matrices!
  // =================================================================
  if (this->use_vibration_mask_ && this->vibration_mask_kdtree_) {
      
      pcl::PointCloud<PointType>::Ptr pruned_scan = std::make_shared<pcl::PointCloud<PointType>>();
      int pruned_count = 0;
      
      // 1.0 meter search radius
      float tau_sq = 1.0f * 1.0f; 

      Eigen::Matrix4f T_local_to_global = this->T_prior * this->extrinsics.baselink2lidar_T;

      for (const auto& pt : this->current_scan->points) {
          if (!std::isfinite(pt.x)) continue;

          Eigen::Vector4f pt_global = T_local_to_global * pt.getVector4fMap();
          PointType p_g;
          p_g.x = pt_global(0); p_g.y = pt_global(1); p_g.z = pt_global(2);

          std::vector<int> indices(1);
          std::vector<float> sq_dists(1);

          if (this->vibration_mask_kdtree_->nearestKSearch(p_g, 1, indices, sq_dists) > 0) {
              if (sq_dists[0] <= tau_sq) {
                  pruned_count++;
                  continue; // DITCH THE POINT! Do not save it.
              }
          }
          // If it's rigid, keep it
          pruned_scan->push_back(pt); 
      }
      
      // Overwrite the scan with the smaller, pruned cloud
      this->current_scan = pruned_scan;
      
      RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
         "[Pruning] Dropped " << pruned_count << " vibrating points to save CPU.");
  }

  // Hand the scan to GICP. 
  // Because the cloud is now smaller, this function will execute MUCH faster!
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void rls::OdomNode::initialize() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }
  this->rls_initialized = true;
}


/*
  Image callback
*/
void rls::OdomNode::callbackImage(const sensor_msgs::msg::Image::SharedPtr img) {
  // RCLCPP_ERROR(this->get_logger(), "Enter to Image node");
  try {
    std::unique_lock<std::mutex> lock(this->mtx_img);

    // Validate message
    if (!img){
      RCLCPP_WARN(this->get_logger(), "Received null image pointer");
      return;
    }

    double msg_header_time;
    try {
      msg_header_time = rclcpp::Time(img->header.stamp).seconds();
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Invalid header timestamp");
      return;
    }

    // Check for time consistency
    if (abs(msg_header_time - this->last_timestamp_img) < 0.001) return;
    if (msg_header_time < this->last_timestamp_img) {
      RCLCPP_ERROR(this->get_logger(), "image loop back.");
      RCLCPP_ERROR(this->get_logger(), "|message header time: %f |last image time: %f", 
                  msg_header_time, this->last_timestamp_img);
      return;
    }

    // Check time jump
    if (msg_header_time - last_timestamp_img < 0.02) {
      RCLCPP_WARN(this->get_logger(), "Image need Jumps: %6f", msg_header_time);
      return;
    }

    if (!this->rls_initialized) {
      return;
    }

    // Convert and store image
    cv::Mat img_curr;
    cv_bridge::CvImagePtr cv_ptr;
    try {
      img_curr = getImageFromMsg(img);
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert image message");
      return;
    }

    // Limit buffer size 
    if (this->img_buffer_.size() > 50) {
      this->img_buffer_.pop_front();
      this->img_time_buffer_.pop_front();
    }
    this->img_buffer_.push_back(img_curr);
    this->img_time_buffer_.push_back(msg_header_time);
    last_timestamp_img = msg_header_time;

    try {
      cv_ptr   = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert image image bridge");
      return;
    }

    // Initialize for optical flow
    cv::Mat curr_frame, curr_frame_color;
    cv::cvtColor(cv_ptr->image, curr_frame, cv::COLOR_BGR2GRAY);
    cv_ptr->image.copyTo(curr_frame_color);
    
    // Compute optical flow
    if (!this->prev_frame.empty() && this->rgb_colorized) {
      std::vector<cv::Point2f> curr_pts;
      std::vector<uchar> status;
      std::vector<float> error;

      // Detect the feature in the previous frame
      cv::goodFeaturesToTrack(this->prev_opt_frame, this->prev_pts, 4500, 0.01, 10);
      cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
      
      // Compute the optical flow
      cv::calcOpticalFlowPyrLK(this->prev_opt_frame, curr_frame, this->prev_pts, curr_pts, status, error, cv::Size(21, 21), 3, criteria);
      
      // RANSAC Outlier Filtering
      std::vector<cv::Point2f> prev_pts_filtered, curr_pts_filtered, prev_pts_filtered_, curr_pts_filtered_;
      std::vector<uchar> ransac_mask;

      // Filter out points with failed flow
      for (size_t i = 0; i < status.size(); i++) {
          if (status[i]) {
              prev_pts_filtered.push_back(prev_pts[i]);
              curr_pts_filtered.push_back(curr_pts[i]);
          }
      }

      // Apply RANSAC to find inlier and outlier
      if (prev_pts_filtered.size() >= 5) {
          cv::Mat F = cv::findFundamentalMat(
                  prev_pts_filtered, curr_pts_filtered,
                  cv::RANSAC, 0.3, 0.99, ransac_mask
          );
      }

      // Draw only inliers (green) and outliers (red)
      for (size_t i = 0; i < ransac_mask.size(); i++) {
          if (ransac_mask[i]) {  // Inlier
              cv::line(curr_frame_color, prev_pts_filtered[i], curr_pts_filtered[i], 
                      cv::Scalar(0, 255, 0), 1);  // Green line
              cv::circle(curr_frame_color, curr_pts_filtered[i], 3, 
                          cv::Scalar(0, 255, 255), -1);  // yellow dot
              prev_pts_filtered_.push_back(prev_pts_filtered[i]);
              curr_pts_filtered_.push_back(curr_pts_filtered[i]);
          } else {  // Outlier
              cv::circle(curr_frame_color, curr_pts_filtered[i], 3, 
                          cv::Scalar(0, 0, 255), -1);  // Red dot
          }
      }

      // Match the point cloud to the feature points
      this->greedyMatching(this->prev_projected_pts, prev_pts_filtered, 3, this->matches);
      if (this->visual_initialization && this->valid_colorized_points->size() > 0 && this->matches.size()){
          // Create filtered cloud and store the original indices
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
          std::vector<int> cloud_idxs, frame_idxs, dist_;
          filtered_cloud->reserve(this->matches.size());
          cloud_idxs.reserve(this->matches.size());
          dist_.reserve(this->matches.size());
          frame_idxs.reserve(this->matches.size());

          for (const auto& m : this->matches) {
            filtered_cloud->push_back(this->prev_colorized_points->points[m.queryIdx]);
            dist_.push_back(m.distance);
            cloud_idxs.push_back(m.queryIdx);
            frame_idxs.push_back(m.trainIdx);
          }

          // Calculate the total observation obtical flow
          float total_dist = 0.0f;
          int valid_pts = 0;

          for (const auto& idx : frame_idxs) {
            if (idx < prev_pts_filtered.size()) {  // Ensure the index is valid
                float dx = curr_pts_filtered[idx].x - prev_pts_filtered[idx].x;
                float dy = curr_pts_filtered[idx].y - prev_pts_filtered[idx].y;
                float distance = std::sqrt(dx * dx + dy * dy);  // Euclidean distance
                
                total_dist += distance;
                valid_pts++;
            }
          }

          // Create output cloud as shared pointer
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
          // Transform the cloud - pass dereferenced pointers
          pcl::transformPointCloud(*filtered_cloud, *transformed_cloud, 
                                    Eigen::Affine3f(this->T_corr));

          // Find the corresponding optical flows                                  
          std::vector<int> correspondences;
          this->findCorrespondences(transformed_cloud, this->valid_colorized_points, correspondences, cloud_idxs, dist_);

          std::vector<int> corr;
          // RCLCPP_INFO(this->get_logger(), "Start optimization node");
          if (!this->prev_colorized_points || !this->valid_colorized_points || this->prev_colorized_points->empty() || this->valid_colorized_points->empty()) {
            // RCLCPP_WARN(this->get_logger(), "Skipping optimization: Empty input clouds");
            return;
          }
          pcl::PointCloud<pcl::Normal>::Ptr target_normals = this->computeNormals(this->prev_colorized_points);
          // this->optimization(this->prev_colorized_points, this->valid_colorized_points, transformed_cloud, target_normals, prev_pts_filtered, curr_pts_filtered, frame_idxs, cloud_idxs, this->T_corr, total_dist);
        }
      // Check optical flow status
      int num_success = std::count(status.begin(), status.end(), 1);

      // Update the previous frame
      this->prev_opt_frame = curr_frame;
      this->rgb_colorized  = false;
    }

    // Only run this if the SLAM has actually started mapping!
    if (this->visual_initialization || this->rls_initialized) {
        static int frame_skip = 0;
        // Save 1 out of every 10 frames (Adjust if you want more/less overlap)
        if (frame_skip % 1 == 0) { 
            RCLCPP_INFO(this->get_logger(), "Splatting Node:");
            // We pass cv_ptr->image (clean) instead of curr_frame_color (drawn on)
            this->exportForSplatting(cv_ptr->image, msg_header_time);
        }
        frame_skip++;
    }

    // Publish image with optical flow
    this->prev_frame = curr_frame;
    cv_bridge::CvImage cv_image;
    cv_image.image    = curr_frame_color.clone();
    cv_image.encoding = sensor_msgs::image_encodings::BGR8;
    cv_image.header.frame_id = "camera_frame";
    this->pubImage.publish(cv_image.toImageMsg());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(this->get_logger(), "Exception in image callback: %s", e.what());
  } catch (...) {
    RCLCPP_FATAL(this->get_logger(), "Unknown exception in image callback");
  }
}

// // Add this function definition
// void rls::OdomNode::exportForSplatting(const cv::Mat& img, double timestamp) {
//     static int frame_count = 0;
//     std::string output_dir = "/tmp/splat_data/";
//     std::string images_dir = output_dir + "images/";
    
//     // Create directories on the first run
//     if (frame_count == 0) {
//         mkdir(output_dir.c_str(), 0777);
//         mkdir(images_dir.c_str(), 0777);
        
//         std::ofstream json_file(output_dir + "transforms.json");
//         // Calculate horizontal Field of View (FOV)
//         double camera_angle_x = 2.0 * atan(this->camera.width / (2.0 * this->camera.fx));
        
//         json_file << "{\n";
//         json_file << "  \"camera_angle_x\": " << camera_angle_x << ",\n";
//         json_file << "  \"fl_x\": " << this->camera.fx << ",\n";
//         json_file << "  \"fl_y\": " << this->camera.fy << ",\n";
//         json_file << "  \"cx\": " << this->camera.cx << ",\n";
//         json_file << "  \"cy\": " << this->camera.cy << ",\n";
//         json_file << "  \"w\": " << this->camera.width << ",\n";
//         json_file << "  \"h\": " << this->camera.height << ",\n";
//         json_file << "  \"frames\": [\n";
//         json_file.close();
//     }

//     // 1. Save the Image
//     std::string img_name = "frame_" + std::to_string(frame_count) + ".jpg";
//     cv::imwrite(images_dir + img_name, img);

//     // 2. Calculate Camera-to-World Transform (T_wc)
//     // Your code currently has World-to-Camera (T_cw) via R_cw and P_cw
//     Eigen::Matrix3f R_wi = this->state.rot;                                      
//     Eigen::Vector3f P_wi = this->state.p;   
//     Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                         
//     Eigen::Vector3f P_cw = -this->Rcl * R_wi.transpose() * P_wi + this->Pcl;
    
//     // Invert to get Camera-to-World
//     Eigen::Matrix3f R_wc = R_cw.transpose();
//     Eigen::Vector3f P_wc = -R_wc * P_cw;

//     // 3. Append to JSON
//     std::ofstream json_file(output_dir + "transforms.json", std::ios_base::app);
//     if (frame_count > 0) json_file << ",\n";
    
//     json_file << "    {\n";
//     json_file << "      \"file_path\": \"images/" << img_name << "\",\n";
//     json_file << "      \"transform_matrix\": [\n";
//     json_file << "        [" << R_wc(0,0) << ", " << R_wc(0,1) << ", " << R_wc(0,2) << ", " << P_wc(0) << "],\n";
//     json_file << "        [" << R_wc(1,0) << ", " << R_wc(1,1) << ", " << R_wc(1,2) << ", " << P_wc(1) << "],\n";
//     json_file << "        [" << R_wc(2,0) << ", " << R_wc(2,1) << ", " << R_wc(2,2) << ", " << P_wc(2) << "],\n";
//     json_file << "        [0.0, 0.0, 0.0, 1.0]\n";
//     json_file << "      ]\n";
//     json_file << "    }";
//     json_file.close();

//     frame_count++;
// }

void rls::OdomNode::exportForSplatting(const cv::Mat& img, double timestamp) {
    static int frame_count = 0;
    // --- CHANGED DIRECTORY ---
    std::string output_dir = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data/";
    // -------------------------
    
    std::string images_dir = output_dir + "images/";
    std::string json_path = output_dir + "transforms.json";
    
    // Create directories on the first run using a forced system command
    if (frame_count == 0) {
        std::string cmd = "mkdir -p " + images_dir;
        int ret = system(cmd.c_str());
        if (ret != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create directory: %s", images_dir.c_str());
        }
        
        std::ofstream json_init(json_path);
        if (!json_init.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "FATAL: Could not create transforms.json at %s", json_path.c_str());
            return; // Stop if we can't write
        }

        // Calculate horizontal Field of View (FOV)
        double camera_angle_x = 2.0 * atan(this->camera.width / (2.0 * this->camera.fx));
        
        json_init << "{\n";
        json_init << "  \"camera_angle_x\": " << camera_angle_x << ",\n";
        json_init << "  \"fl_x\": " << this->camera.fx << ",\n";
        json_init << "  \"fl_y\": " << this->camera.fy << ",\n";
        json_init << "  \"cx\": " << this->camera.cx << ",\n";
        json_init << "  \"cy\": " << this->camera.cy << ",\n";
        json_init << "  \"w\": " << this->camera.height << ",\n";
        json_init << "  \"h\": " << this->camera.width  << ",\n";
        json_init << "  \"frames\": [\n";
        json_init.close();
        
        RCLCPP_INFO(this->get_logger(), "Successfully initialized Splatting export at %s", output_dir.c_str());
    }

    // 1. Save the Image
    std::string img_name = "frame_" + std::to_string(frame_count) + ".jpg";
    if (!cv::imwrite(images_dir + img_name, img)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to save image: %s", img_name.c_str());
        return;
    }

    // 2. Calculate Camera-to-World Transform (T_wc)
    Eigen::Matrix3f R_wi = this->state.rot;                                      
    Eigen::Vector3f P_wi = this->state.p;   
    Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                         
    Eigen::Vector3f P_cw = -this->Rcl * R_wi.transpose() * P_wi + this->Pcl;
    
    // Invert to get Camera-to-World
    Eigen::Matrix3f R_wc = R_cw.transpose();
    Eigen::Vector3f P_wc = -R_wc * P_cw;

    // 3. Append to JSON
    std::ofstream json_file(json_path, std::ios_base::app);
    if (!json_file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open %s for appending!", json_path.c_str());
        return;
    }

    if (frame_count > 0) json_file << ",\n";
    
    json_file << "    {\n";
    json_file << "      \"file_path\": \"images/" << img_name << "\",\n";
    json_file << "      \"transform_matrix\": [\n";
    json_file << "        [" << R_wc(0,0) << ", " << R_wc(0,1) << ", " << R_wc(0,2) << ", " << P_wc(0) << "],\n";
    json_file << "        [" << R_wc(1,0) << ", " << R_wc(1,1) << ", " << R_wc(1,2) << ", " << P_wc(1) << "],\n";
    json_file << "        [" << R_wc(2,0) << ", " << R_wc(2,1) << ", " << R_wc(2,2) << ", " << P_wc(2) << "],\n";
    json_file << "        [0.0, 0.0, 0.0, 1.0]\n";
    json_file << "      ]\n";
    json_file << "    }";
    json_file.close();

    // Print to terminal every 50 frames so you know it's working
    if (frame_count % 50 == 0) {
        RCLCPP_INFO(this->get_logger(), "Exported %d frames for Splatting...", frame_count);
    }

    frame_count++;
}

/*
    Define motion prior cost
*/
rls::OdomNode::MotionPriorCost::MotionPriorCost (const double* prior_pose, double weight)
                                                  : prior_pose_(prior_pose), weight_(weight) {}

/*
  Define the optical flow cost function
*/
rls::OdomNode::FlowCostFunction::FlowCostFunction (const std::vector<cv::Point2f> &curr_pts_filtered,
                     const std::vector<cv::Point2f> &prev_pts_filtered,
                     const std::vector<int> &frame_idxs,
                     const std::vector<int> &cloud_idxs,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                     const Eigen::Matrix3f& R_cl,
                     const Eigen::Vector3f& P_cl,
                     const Eigen::Matrix3f& R_corr,
                     const Eigen::Vector3f& P_corr,
                     const Eigen::Matrix3f &K,
                     double max_correspondence_dist = 0.5)
                     : curr_pts_(curr_pts_filtered),
                       prev_pts_(prev_pts_filtered),
                       frame_idxs_(frame_idxs),
                       cloud_idxs_(cloud_idxs),
                       source_cloud_(source_cloud),
                       target_cloud_(target_cloud),
                       R_cl_(R_cl), P_cl_(P_cl),
                       R_corr_(R_corr), P_corr_(P_corr),
                       K_(K),
                       max_correspondence_dist_(max_correspondence_dist) {
                       
                       if (!target_cloud_) {
                        throw std::invalid_argument("Target cloud is null");
                       }
                       // Pre-build KD-Tree for target cloud
                       kdtree_.setInputCloud(target_cloud);
}

/*
    Define Point To Point cost function
*/
rls::OdomNode::PointToPointCostFunction::PointToPointCostFunction (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target,
    double max_dist)
    : source_(source), 
      target_(target), 
      max_corr_dist_sq_p2p_(max_dist * max_dist) {
      establishCorrespondencesP2P();
}

void rls::OdomNode::PointToPointCostFunction::establishCorrespondencesP2P() {
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(target_);
    
    correspondences_.resize(source_->size());
    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (size_t i = 0; i < source_->size(); ++i) {
        if (kdtree.nearestKSearch(source_->points[i], 1, indices, distances) > 0) {
            if (distances[0] <= max_corr_dist_sq_p2p_) {
                correspondences_[i] = indices[0];
            } else {
                correspondences_[i] = -1;  // Mark as invalid
            }
        }
    }
}

/*
  Define GICP cost function
*/ 
rls::OdomNode::GICPCostFunction::GICPCostFunction (    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& target_normals,
    double max_corr_dist)
    : source_cloud_(source_cloud),
      target_cloud_(target_cloud),
      target_normals_(target_normals),
      max_corr_dist_sq_(max_corr_dist * max_corr_dist) {
    establishCorrespondences();
}

void rls::OdomNode::GICPCostFunction::establishCorrespondences() {
    correspondences_.clear();
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(target_cloud_);

    for (size_t i = 0; i < source_cloud_->size(); ++i) {
        const auto& point = source_cloud_->points[i];
        if (!pcl::isFinite(point)) continue;
        
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        
        if (kdtree.nearestKSearch(point, 1, indices, distances) > 0) {
            if (distances[0] <= max_corr_dist_sq_) {
                // Additional check for valid normal
                if (pcl::isFinite(target_normals_->points[indices[0]])) {
                    correspondences_.push_back(indices[0]);
                }
            }
        }
    }
}


/*
  Optimization function
*/
void rls::OdomNode::optimization(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &transformed_cloud,
                                  const pcl::PointCloud<pcl::Normal>::Ptr &target_normals,
                                  const std::vector<cv::Point2f> &prev_pts,
                                  const std::vector<cv::Point2f> &curr_pts,
                                  const std::vector<int> &frame_idxs,
                                  const std::vector<int> &cloud_idxs,
                                  const Eigen::Matrix4f &T_corr,
                                  int total_dist) {

  // Input Validation                                     
  if (!source_cloud || !target_cloud || source_cloud->empty() || target_cloud->empty()) {
    RCLCPP_WARN(this->get_logger(), "Skipping optimization: Empty input clouds");
    return;
  }
                                    
  // Convert quaternion to angle-axis for Ceres
  Eigen::AngleAxisf init_rotation(this->state.q);
  double pose[6] = {
      init_rotation.angle() * init_rotation.axis().x(),
      init_rotation.angle() * init_rotation.axis().y(),
      init_rotation.angle() * init_rotation.axis().z(),
      this->state.p.x(),  // position from state
      this->state.p.y(),
      this->state.p.z()
  };

  // Store prior motion
  double prior_pose[6];
  std::copy(pose, pose+6, prior_pose);

  // Extract the correlation transform
  Eigen::Matrix3f R_corr = T_corr.block<3, 3>(0, 0);
  Eigen::Vector3f P_corr = T_corr.block<3, 1>(0, 3);

  // Weight for optimization
  const double gicp_weight = 1.0;
  const double p2p_weight  = 1.0;
  const double flow_weight = 0.01;
  const double motion_prior_weight = 0.8;

  ceres::Problem problem;
  bool valid_residuals = false;
  
  if (this->align_paramas_) {
    // // Add GICP residuals
    auto* gicp_cost = new GICPCostFunction(target_cloud, source_cloud, target_normals, this->gicp_max_corr_dist_);
    const size_t gicp_correspondences = gicp_cost->getNumCorrespondences();

    if (gicp_correspondences >= 20) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<GICPCostFunction, ceres::DYNAMIC, 6>(
              gicp_cost, gicp_correspondences),
          new ceres::ScaledLoss(new ceres::HuberLoss(0.1), 1.0, ceres::TAKE_OWNERSHIP),
          pose
          );
          valid_residuals = true;
          RCLCPP_DEBUG(this->get_logger(), "Added GICP with %zu correspondences", gicp_correspondences);
      } else {
          delete gicp_cost;
          RCLCPP_WARN(this->get_logger(), "Insufficient GICP correspondences: %zu", gicp_correspondences);
    }

    if (valid_residuals) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<MotionPriorCost, 6, 6>(
              new MotionPriorCost(prior_pose, motion_prior_weight)),
          nullptr,
          pose
      );
    } else {
      RCLCPP_ERROR(this->get_logger(), "No valid residuals for optimization");
      return;
    }

    auto flow_cost = std::make_unique<FlowCostFunction>(
        curr_pts, prev_pts, frame_idxs, cloud_idxs,
        source_cloud, target_cloud,
        this->Rcl, this->Pcl,
        R_corr, P_corr,
        this->camera.intrinsic_matrix,
        0.5);

    std::vector<double> test_residuals(2 * frame_idxs.size());
    if ((*flow_cost)(pose, test_residuals.data())) {
      int residual_count = flow_cost->getValidCount();
      if (residual_count > 50) {
        RCLCPP_INFO(this->get_logger(), "Add Flow cost");
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<FlowCostFunction, ceres::DYNAMIC, 6>(
                flow_cost.release(), 2 * frame_idxs.size()),
            new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5),
                flow_weight,
                ceres::TAKE_OWNERSHIP),
            pose
        );
        RCLCPP_INFO(this->get_logger(), "Successfully added flow residuals (%d points)", residual_count);
      } else {
        RCLCPP_WARN(this->get_logger(), "Skipped flow residuals: insufficient valid correspondences (%d)", residual_count);
      }
    } else {
      // RCLCPP_WARN(this->get_logger(), "Flow residuals rejected during pre-evaluation");
    }
  } else {
    // Add Point to Point residuals
    auto* p2p_cost  = new PointToPointCostFunction(target_cloud, source_cloud, this->gicp_max_corr_dist_);
    const size_t p2p_correspondences = p2p_cost->getNumCorrespondences();
    if (p2p_correspondences >= 20) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<PointToPointCostFunction, ceres::DYNAMIC, 6>(
              p2p_cost, p2p_correspondences),
          new ceres::ScaledLoss(new ceres::HuberLoss(0.1), 1.0, ceres::TAKE_OWNERSHIP),
          pose
      );
      valid_residuals = true;
      RCLCPP_DEBUG(this->get_logger(), "Added P2P with %zu correspondences", p2p_correspondences);
    } else {
      delete p2p_cost;
      RCLCPP_WARN(this->get_logger(), "Insufficient P2P correspondences: %zu", p2p_correspondences);
    }
    std::cout << "|Size of the clouds : " << " Target size = " << target_cloud->size() << " |Source size =  " << source_cloud->size() << std::endl;
    
    // // Add GICP residuals
    auto* gicp_cost = new GICPCostFunction(target_cloud, source_cloud, target_normals, this->gicp_max_corr_dist_);
    const size_t gicp_correspondences = gicp_cost->getNumCorrespondences();

    if (gicp_correspondences >= 20) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<GICPCostFunction, ceres::DYNAMIC, 6>(
              gicp_cost, gicp_correspondences),
          new ceres::ScaledLoss(new ceres::HuberLoss(0.1), 1.0, ceres::TAKE_OWNERSHIP),
          pose
          );
          valid_residuals = true;
          RCLCPP_DEBUG(this->get_logger(), "Added GICP with %zu correspondences", gicp_correspondences);
      } else {
          delete gicp_cost;
          RCLCPP_WARN(this->get_logger(), "Insufficient GICP correspondences: %zu", gicp_correspondences);
    }

    if (valid_residuals) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<MotionPriorCost, 6, 6>(
              new MotionPriorCost(prior_pose, motion_prior_weight)),
          nullptr,
          pose
      );
    } else {
      RCLCPP_ERROR(this->get_logger(), "No valid residuals for optimization");
      return;
    }

    auto flow_cost = std::make_unique<FlowCostFunction>(
        curr_pts, prev_pts, frame_idxs, cloud_idxs,
        source_cloud, target_cloud,
        this->Rcl, this->Pcl,
        R_corr, P_corr,
        this->camera.intrinsic_matrix,
        0.5);

    std::vector<double> test_residuals(2 * frame_idxs.size());
    if ((*flow_cost)(pose, test_residuals.data())) {
      int residual_count = flow_cost->getValidCount();
      if (residual_count > 50) {
        RCLCPP_INFO(this->get_logger(), "Add Flow cost");
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<FlowCostFunction, ceres::DYNAMIC, 6>(
                flow_cost.release(), 2 * frame_idxs.size()),
            new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5),
                flow_weight,
                ceres::TAKE_OWNERSHIP),
            pose
        );
        RCLCPP_INFO(this->get_logger(), "Successfully added flow residuals (%d points)", residual_count);
      } else {
        RCLCPP_WARN(this->get_logger(), "Skipped flow residuals: insufficient valid correspondences (%d)", residual_count);
      }
    } else {
      // RCLCPP_WARN(this->get_logger(), "Flow residuals rejected during pre-evaluation");
    }
  }

  // Configure Solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // Best for large problems
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
  options.num_threads = 2;
  options.max_num_iterations = 20;
  options.function_tolerance = 1e-6;
  options.parameter_tolerance = 1e-8;
  options.minimizer_progress_to_stdout = true;  // Debug output
  options.num_threads = std::thread::hardware_concurrency();

  // Trust region strategy (critical for hybrid residuals)
  options.update_state_every_iteration = true;

// Check all parameter blocks
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  for (auto* block : parameter_blocks) {
      for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(block[i])) {
          RCLCPP_ERROR(this->get_logger(), "Invalid parameter at position %d: %f", i, block[i]);
        }
      }
  }
  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (!summary.IsSolutionUsable()) {
    RCLCPP_ERROR(this->get_logger(), "Optimization failed: %s", 
                  summary.message.c_str());
  }
  // Print full report
  RCLCPP_INFO_STREAM(this->get_logger(), summary.BriefReport());

  // Convert optimized Ceres pose back to state
  Eigen::Vector3d angle_axis(pose[0], pose[1], pose[2]);
  double angle = angle_axis.norm();

  if (angle > 1e-10) {
      Eigen::Vector3d axis = angle_axis / angle;
      Eigen::Quaternionf q = Eigen::Quaternionf(
          Eigen::AngleAxisf(static_cast<float>(angle), 
                          axis.cast<float>()));
      this->state.q = q;
  } else {
      this->state.q = Eigen::Quaternionf::Identity();
  }

  Eigen::Vector3f p = Eigen::Vector3f(
      static_cast<float>(pose[3]),
      static_cast<float>(pose[4]),
      static_cast<float>(pose[5]));
      this->state.p = 0.8 * this->state.p + 0.2 * p;
  RCLCPP_INFO(this->get_logger(), "Residual blocks: %d", summary.num_residual_blocks);
  RCLCPP_INFO(this->get_logger(), "Residuals: %d", summary.num_residuals);
}



/*
  Compute target normals
*/
pcl::PointCloud<pcl::Normal>::Ptr rls::OdomNode::computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  ne.setSearchMethod(tree);

  // Use KNN (K-nearest neighbor)
  // const int k = 5; 
  // ne.setKSearch(k);
  // ne.compute(*normals);

  ne.setRadiusSearch(0.2);
  ne.compute(*normals);

  for (auto& normal : *normals) {
    if (!pcl::isFinite(normal)) {
      normal.normal_x = normal.normal_y = normal.normal_z = 0;
      normal.curvature = 0;
    }
  }
  return normals;
}

/*
  Calculate flow cost of the estimation
*/
void rls::OdomNode::calcFlowEstCost(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src_cloud,
                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &tgt_cloud,
                       const std::vector<cv::Point2f> &curr_pts,
                       const std::vector<cv::Point2f> &prev_pts,
                       const std::vector<int> &frame_idxs) {
  
  // Input validation
  if (!src_cloud || src_cloud->empty() || !tgt_cloud || tgt_cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Invalid input clouds");
    return;
  }
  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(tgt_cloud);
  std::vector<int> indices(1);
  std::vector<float> distances(1);

  Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
  Eigen::Vector3f P_wi = this->state.p;   
  Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                        // Camera2world
  Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

  float opt_est = 0.0f;

  for (size_t i = 0; i < src_cloud->size(); i++) {
      // Skip NaN points
      if (!pcl::isFinite(src_cloud->points[i])) {
          continue;
      }

      if (kdtree.nearestKSearch(src_cloud->points[i], 1, indices, distances) == 0) {
          continue;  // No neighbors found
      }

      if (distances[0] < 0.2 && indices[0] >= 0 && indices[0] < tgt_cloud->size()) {
        int tgt_idx = indices[0];
        if (tgt_idx < 0 || tgt_idx >= tgt_cloud->size()) continue;
        if (!pcl::isFinite(tgt_cloud->points[tgt_idx])) continue;

        // Validate frame index
        int obs_idx;
        try {
              obs_idx = frame_idxs.at(i);
              if (obs_idx < 0 || obs_idx >= prev_pts.size() || obs_idx >= curr_pts.size()) {
                  continue;
              }
        } catch (const std::out_of_range& e) {
              continue;
        }
        
        Eigen::Vector3f pt(tgt_cloud->points[tgt_idx].x, 
                           tgt_cloud->points[tgt_idx].y,
                           tgt_cloud->points[tgt_idx].z);
        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;            
        
        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);

        float dx = u - prev_pts[obs_idx].x;
        float dy = v - prev_pts[obs_idx].y;
        opt_est += std::sqrt(dx * dx + dy * dy);
      }
  }
  this->optical_est = opt_est;
}

/*
  Find correspondences function
*/
void rls::OdomNode::findCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                                         const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cloud,
                                         std::vector<int> &correspondences,
                                         const std::vector<int> &cloud_idxs,
                                         const std::vector<int> &dist_) {

  // Input validation
  if (!source_cloud || source_cloud->empty() || !target_cloud || target_cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Invalid input clouds");
    correspondences.clear();
    return;
  }

  if (!cloud_idxs.empty() && cloud_idxs.size() != source_cloud->size()) {
    RCLCPP_ERROR(this->get_logger(), "cloud_idxs size mismatch with source cloud");
    correspondences.clear();
    return;
  }


  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(target_cloud);
  std::vector<int> indices(1);
  std::vector<float> distances(1);

  correspondences.resize(source_cloud->size(), -1);
  int count = 0;
  int count_opt = 0;
  
  float opt_est = 0.0f;
  float opt_obs = 0.0f;

  for (size_t i = 0; i < source_cloud->size(); ++i) {

      // Skip NaN points
      if (!pcl::isFinite(source_cloud->points[i])) {
          continue;
      }

      /*
        If Option 2 uncomment these lines below:
      */
      if (kdtree.nearestKSearch(source_cloud->points[i], 1, indices, distances) == 0) {
          continue;  // No neighbors found
      }

      // Validate the found index
      if (distances[0] < 0.2 && indices[0] >= 0 && indices[0] < target_cloud->size()) {
          correspondences[i] = indices[0];
          const auto& src_pt = source_cloud->points[i];
          const auto& tgt_pt = target_cloud->points[indices[0]];
          
          int src_idx = cloud_idxs.empty() ? i : cloud_idxs[i];
          count++;

          // Extract the corresponding pixels between two frames
          if (src_idx >= 0 && src_idx < this->prev_projected_pts.size() && 
              indices[0] >= 0 && indices[0] < this->projected_pts.size()) {
              float dx = this->prev_projected_pts[src_idx].x - this->projected_pts[indices[0]].x;
              float dy = this->prev_projected_pts[src_idx].y - this->projected_pts[indices[0]].y;
              opt_est += std::sqrt(dx * dx + dy * dy);
              count_opt++;
          } else {
              RCLCPP_WARN(this->get_logger(), "Projected points index out of bounds (src: %d/%zu, tgt: %d/%zu)",
                          src_idx, this->prev_projected_pts.size(), 
                          indices[0], this->projected_pts.size());
          }         
      }
      opt_obs += dist_[i];
  }
  this->optical_est = opt_est;
}

/*
  Matching Point cloud to image features
*/
void rls::OdomNode::greedyMatching(const std::vector<cv::Point2f> &projected_pts, 
                                    const std::vector<cv::Point2f> &featured_pts,
                                    float max_dist_thres,
                                    std::vector<cv::DMatch> &matches) {
  // RCLCPP_ERROR(this->get_logger(), "Enter to matching node");
  // this->matches.clear();
  matches.clear();
  std::vector<bool> feature_matched(featured_pts.size(), false);

  for (size_t i = 0; i < projected_pts.size(); ++i) {
    float min_dist = std::numeric_limits<float>::max();
    int best_j = -1;

    for (size_t j = 0; j < featured_pts.size(); ++j) {
      if (feature_matched[j]) continue;

      float dx = projected_pts[i].x - featured_pts[j].x;
      float dy = projected_pts[i].y - featured_pts[j].y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist < min_dist && dist < max_dist_thres) {
        min_dist = dist;
        best_j   = j;
      }
    }

    if (best_j != -1) {
      this->matches.emplace_back(i, best_j, min_dist);
      feature_matched[best_j] = true;
    }
  }

  int count = 0;

}

/*
  Sub-Function for Image callback
*/
cv::Mat rls::OdomNode::getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
    cv::Mat img;
    try {
        // Convert ROS2 image message to OpenCV Mat
        img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv::Mat();  // Return empty Mat on error
    }
    return img;
}

// /*
//   Projecting LiDAR point to image
// */
// void rls::OdomNode::projectLidarToImage(const cv::Mat &img) {
  
//   // 1. Initialize PCL point clouds
//   pcl::PointCloud<PointType>::Ptr voxelized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::PointCloud<PointType>::Ptr deskewed_scan_t_visual = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::transformPointCloud(*this->deskewed_scan, *deskewed_scan_t_visual, this->T_corr );

//   // 2. Voxel Grid Filtering
//   pcl::VoxelGrid<PointType> voxel_filter;
//   voxel_filter.setInputCloud(deskewed_scan_t_visual);     // Input cloud (transformed)
//   voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f);       // Voxel size (adjust as needed)
//   voxel_filter.filter(*voxelized_cloud);                  // Output: voxelized_cloud
//   // RCLCPP_DEBUG(this->get_logger(), "Voxelized cloud size: %zu", voxelized_cloud->size());
//   this->valid_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   this->valid_colorized_points->points.reserve(voxelized_cloud->size());
//   this->prev_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   this->prev_colorized_points->points.reserve(voxelized_cloud->size());
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudWorldRGB = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   laserCloudWorldRGB->reserve(voxelized_cloud->size());

//   // Exrtact the state orientation
//   Eigen::Quaternionf q = this->state.q.normalized();
//   this->state.rot      = q.toRotationMatrix();
//   // Calculate the transformation between coordinates
//   Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
//   Eigen::Vector3f P_wi = this->state.p;   
//   Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                        // Camera2world
//   Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

//   if (img.empty()) {
//   RCLCPP_WARN(this->get_logger(), "Image is empty, skipping projection.");
//   return;
//   }

//   // --- NEW: 1. Blur the image to remove camera ISO grain ---
//   cv::Mat smooth_img;
//   cv::GaussianBlur(img, smooth_img, cv::Size(3, 3), 0);
//   // ---------------------------------------------

//   this->projected_pts.clear();                                                // Clear the previous points
//   this->projected_pts.reserve(voxelized_cloud->size());                       // Pre-allocate memory
//   this->prev_projected_pts.reserve(voxelized_cloud->size());

//   if (this->visual_initialization) {

//     for (int i = 0; i < voxelized_cloud->size(); i++) {
//     // for (int i = 0; i < this->deskewed_scan->size(); i++) {
//         Eigen::Vector3f pt(voxelized_cloud->points[i].x,
//               voxelized_cloud->points[i].y,
//               voxelized_cloud->points[i].z);

//         Eigen::Vector3f   pc =  R_cw * pt + P_cw;
//         Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

//         // Normalize to get pixel coordinates
//         float u = image_points(0) / image_points(2);
//         float v = image_points(1) / image_points(2);

//         if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
//             continue;
//         }

//         if (u >= 0 && u < (this->camera.width - 1) && v >= 0 && v < (this->camera.height - 1) && image_points(2) > 0) {
//           // RCLCPP_WARN(this->get_logger(), "Valid image pointer");
//           this->projected_pts.emplace_back(u, v);
//           int u_floor = static_cast<int>(u);
//           int v_floor = static_cast<int>(v);
//           float u_frac = u - u_floor;
//           float v_frac = v - v_floor;

//           // Get the 4 neighboring pixels
//           cv::Vec3b p00 = img.at<cv::Vec3b>(v_floor, u_floor);
//           cv::Vec3b p01 = img.at<cv::Vec3b>(v_floor, u_floor + 1);
//           cv::Vec3b p10 = img.at<cv::Vec3b>(v_floor + 1, u_floor);
//           cv::Vec3b p11 = img.at<cv::Vec3b>(v_floor + 1, u_floor + 1);

//           // Interpolate each channel (BGR) separately
//           cv::Vec3b color;
//           for (int i = 0; i < 3; i++) {
//               float channel_val = 
//                   (1 - u_frac) * (1 - v_frac) * p00[i] +
//                   u_frac * (1 - v_frac) * p01[i] +
//                   (1 - u_frac) * v_frac * p10[i] +
//                   u_frac * v_frac * p11[i];
//               color[i] = static_cast<unsigned char>(std::round(channel_val));
//           }
//           pcl::PointXYZRGB colored_point;
//           colored_point.x = pt(0);
//           colored_point.y = pt(1);
//           colored_point.z = pt(2);
//           colored_point.r = color[2];
//           colored_point.g = color[1];
//           colored_point.b = color[0];
//           // // Add point to rgb point cloud
//           this->valid_colorized_points->points.push_back(colored_point);
//           laserCloudWorldRGB->push_back(colored_point);
//         }
//       }
//       this->valid_point_flag = true;
//       this->cloud_rgb_buffer.push_back(this->valid_colorized_points);
//       this->pts_buffer.push_back(this->projected_pts);
//   } else {
//     for (int i = 0; i < voxelized_cloud->size(); i++) {
//         Eigen::Vector3f pt(voxelized_cloud->points[i].x,
//               voxelized_cloud->points[i].y,
//               voxelized_cloud->points[i].z);

//         Eigen::Vector3f   pc =  R_cw * pt + P_cw;
//         Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

//         // Normalize to get pixel coordinates
//         float u = image_points(0) / image_points(2);
//         float v = image_points(1) / image_points(2);
        
//         if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
//             continue;
//         }

//         if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height && image_points(2) > 0) {
//           this->prev_projected_pts.emplace_back(u, v);
//           cv::Vec3b color = img.at<cv::Vec3b>(v, u);

//           pcl::PointXYZRGB colored_point;
//           colored_point.x = pt(0);
//           colored_point.y = pt(1);
//           colored_point.z = pt(2);
//           colored_point.r = color[2];
//           colored_point.g = color[1];
//           colored_point.b = color[0];
//           // // Add point to rgb point cloud
//           this->prev_colorized_points->points.push_back(colored_point);
//           laserCloudWorldRGB->push_back(colored_point);
//         }
//     }
//     this->prev_opt_frame = this->prev_frame;
//   }

//   if (laserCloudWorldRGB->size() > 0) {
//     this->visual_initialization = true;
//   }

//   if (this->valid_colorized_points->size() > 0 && this->projected_pts.size() > 0) {
//     this->prev_colorized_points = this->valid_colorized_points;
//     this->prev_projected_pts    = this->projected_pts;
//   }

//   if (this->map3d_params_) {
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_pts_t = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//     sensor_msgs::msg::PointCloud2 rgb_pc;
//     pcl::toROSMsg(*laserCloudWorldRGB, rgb_pc);
//     rgb_pc.header.stamp = this->scan_header_stamp;
//     rgb_pc.header.frame_id = this->odom_frame;
//     this->deskewed_pub->publish(rgb_pc);
//     this->rgb_colorized = true;
//   }

// }

// /*
//   Projecting LiDAR point to image (With Z-Buffer Occlusion & Noise Filtering)
// */
// void rls::OdomNode::projectLidarToImage(const cv::Mat &img) {
  
//   // 1. Initialize PCL point clouds
//   pcl::PointCloud<PointType>::Ptr voxelized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::PointCloud<PointType>::Ptr deskewed_scan_t_visual = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::transformPointCloud(*this->deskewed_scan, *deskewed_scan_t_visual, this->T_corr );

//   // 2. Voxel Grid Filtering
//   pcl::VoxelGrid<PointType> voxel_filter;
//   voxel_filter.setInputCloud(deskewed_scan_t_visual);     // Input cloud (transformed)
//   voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f);       // Voxel size (adjust as needed)
//   voxel_filter.filter(*voxelized_cloud);                  // Output: voxelized_cloud
  
//   this->valid_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   this->valid_colorized_points->points.reserve(voxelized_cloud->size());
//   this->prev_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   this->prev_colorized_points->points.reserve(voxelized_cloud->size());
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudWorldRGB = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   laserCloudWorldRGB->reserve(voxelized_cloud->size());

//   // Extract the state orientation
//   Eigen::Quaternionf q = this->state.q.normalized();
//   this->state.rot      = q.toRotationMatrix();
  
//   // Calculate the transformation between coordinates
//   Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
//   Eigen::Vector3f P_wi = this->state.p;   
//   Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                         // Camera2world
//   Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

//   if (img.empty()) {
//       RCLCPP_WARN(this->get_logger(), "Image is empty, skipping projection.");
//       return;
//   }

//   // =========================================================================
//   // [NEW] FILTER 1: Gaussian Blur to remove ISO Camera Grain / White Noise
//   // =========================================================================
//   cv::Mat smooth_img;
//   cv::GaussianBlur(img, smooth_img, cv::Size(3, 3), 0);

//   this->projected_pts.clear();                                              // Clear the previous points
//   this->projected_pts.reserve(voxelized_cloud->size());                     // Pre-allocate memory
//   this->prev_projected_pts.reserve(voxelized_cloud->size());

//   // =========================================================================
//   // [NEW] FILTER 2: Build a Z-Buffer (Depth Map) for Occlusion Culling
//   // =========================================================================
//   cv::Mat depth_buffer(this->camera.height, this->camera.width, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));

//   // PASS 1: Find the closest depth value for every pixel
//   for (int i = 0; i < voxelized_cloud->size(); i++) {
//       Eigen::Vector3f pt(voxelized_cloud->points[i].x, voxelized_cloud->points[i].y, voxelized_cloud->points[i].z);
//       Eigen::Vector3f pc = R_cw * pt + P_cw;
      
//       if (pc.z() <= 0) continue; // Skip points behind camera

//       Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;
//       int u = static_cast<int>(std::round(image_points(0) / image_points(2)));
//       int v = static_cast<int>(std::round(image_points(1) / image_points(2)));

//       // If point is in frame, update the depth buffer with the closest Z value
//       if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height) {
//           // Expand footprint slightly (3x3 pixels) to account for LiDAR sparsity preventing holes
//           for (int du = -1; du <= 1; du++) {
//               for (int dv = -1; dv <= 1; dv++) {
//                   int nu = u + du, nv = v + dv;
//                   if (nu >= 0 && nu < this->camera.width && nv >= 0 && nv < this->camera.height) {
//                       if (pc.z() < depth_buffer.at<float>(nv, nu)) {
//                           depth_buffer.at<float>(nv, nu) = pc.z();
//                       }
//                   }
//               }
//           }
//       }
//   }

//   // =========================================================================
//   // PASS 2: Project and Colorize (With Occlusion Rejection)
//   // =========================================================================
//   if (this->visual_initialization) {

//     for (int i = 0; i < voxelized_cloud->size(); i++) {
//         Eigen::Vector3f pt(voxelized_cloud->points[i].x,
//               voxelized_cloud->points[i].y,
//               voxelized_cloud->points[i].z);

//         Eigen::Vector3f   pc =  R_cw * pt + P_cw;
//         if (pc.z() <= 0) continue;

//         Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

//         // Normalize to get pixel coordinates
//         float u = image_points(0) / image_points(2);
//         float v = image_points(1) / image_points(2);

//         if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height) {
//             continue;
//         }

//         if (u >= 0 && u < (this->camera.width - 1) && v >= 0 && v < (this->camera.height - 1)) {
          
//           // --- OCCLUSION CHECK ---
//           int u_int = static_cast<int>(std::round(u));
//           int v_int = static_cast<int>(std::round(v));
//           float closest_depth = depth_buffer.at<float>(v_int, u_int);
          
//           // If this point is 0.3 meters further away than the closest point
//           // recorded at this pixel, it's a background point grabbing a foreground color! Skip it.
//           if (pc.z() > closest_depth + 0.3f) {
//               continue; 
//           }
//           // -----------------------

//           this->projected_pts.emplace_back(u, v);
//           int u_floor = static_cast<int>(u);
//           int v_floor = static_cast<int>(v);
//           float u_frac = u - u_floor;
//           float v_frac = v - v_floor;

//           // Get the 4 neighboring pixels from the SMOOTHED image
//           cv::Vec3b p00 = smooth_img.at<cv::Vec3b>(v_floor, u_floor);
//           cv::Vec3b p01 = smooth_img.at<cv::Vec3b>(v_floor, u_floor + 1);
//           cv::Vec3b p10 = smooth_img.at<cv::Vec3b>(v_floor + 1, u_floor);
//           cv::Vec3b p11 = smooth_img.at<cv::Vec3b>(v_floor + 1, u_floor + 1);

//           // Interpolate each channel (BGR) separately
//           cv::Vec3b color;
//           for (int c = 0; c < 3; c++) {
//               float channel_val = 
//                   (1 - u_frac) * (1 - v_frac) * p00[c] +
//                   u_frac * (1 - v_frac) * p01[c] +
//                   (1 - u_frac) * v_frac * p10[c] +
//                   u_frac * v_frac * p11[c];
//               color[c] = static_cast<unsigned char>(std::round(channel_val));
//           }

//           pcl::PointXYZRGB colored_point;
//           colored_point.x = pt(0);
//           colored_point.y = pt(1);
//           colored_point.z = pt(2);
//           colored_point.r = color[2];
//           colored_point.g = color[1];
//           colored_point.b = color[0];
          
//           // Add point to rgb point cloud
//           this->valid_colorized_points->points.push_back(colored_point);
//           laserCloudWorldRGB->push_back(colored_point);
//         }
//       }
//       this->valid_point_flag = true;
//       this->cloud_rgb_buffer.push_back(this->valid_colorized_points);
//       this->pts_buffer.push_back(this->projected_pts);

//   } else {
    
//     // First frame initialization logic
//     for (int i = 0; i < voxelized_cloud->size(); i++) {
//         Eigen::Vector3f pt(voxelized_cloud->points[i].x,
//               voxelized_cloud->points[i].y,
//               voxelized_cloud->points[i].z);

//         Eigen::Vector3f   pc =  R_cw * pt + P_cw;
//         if (pc.z() <= 0) continue;

//         Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

//         // Normalize to get pixel coordinates
//         float u = image_points(0) / image_points(2);
//         float v = image_points(1) / image_points(2);
        
//         if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height) {
//             continue;
//         }

//         if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height) {
          
//           // --- OCCLUSION CHECK ---
//           int u_int = static_cast<int>(std::round(u));
//           int v_int = static_cast<int>(std::round(v));
//           float closest_depth = depth_buffer.at<float>(v_int, u_int);
          
//           if (pc.z() > closest_depth + 0.3f) {
//               continue; 
//           }
//           // -----------------------

//           this->prev_projected_pts.emplace_back(u, v);
          
//           // Use the smoothed image here as well
//           cv::Vec3b color = smooth_img.at<cv::Vec3b>(v_int, u_int);

//           pcl::PointXYZRGB colored_point;
//           colored_point.x = pt(0);
//           colored_point.y = pt(1);
//           colored_point.z = pt(2);
//           colored_point.r = color[2];
//           colored_point.g = color[1];
//           colored_point.b = color[0];
          
//           // Add point to rgb point cloud
//           this->prev_colorized_points->points.push_back(colored_point);
//           laserCloudWorldRGB->push_back(colored_point);
//         }
//     }
//     this->prev_opt_frame = this->prev_frame;
//   }

//   if (laserCloudWorldRGB->size() > 0) {
//     this->visual_initialization = true;
//   }

//   if (this->valid_colorized_points->size() > 0 && this->projected_pts.size() > 0) {
//     this->prev_colorized_points = this->valid_colorized_points;
//     this->prev_projected_pts    = this->projected_pts;
//   }

//   if (this->map3d_params_) {
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_pts_t = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//     sensor_msgs::msg::PointCloud2 rgb_pc;
//     pcl::toROSMsg(*laserCloudWorldRGB, rgb_pc);
//     rgb_pc.header.stamp = this->scan_header_stamp;
//     rgb_pc.header.frame_id = this->odom_frame;
//     this->deskewed_pub->publish(rgb_pc);
//     this->rgb_colorized = true;

//     // ==========================================================
//     static int cloud_save_count = 0;
//     if (cloud_save_count % 50 == 0) {
//         std::string save_dir = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data/pointclouds/";
        
//         // Fix warning: check system return value
//         std::string mkdir_cmd = "mkdir -p " + save_dir;
//         if (system(mkdir_cmd.c_str()) != 0) {
//             RCLCPP_WARN(this->get_logger(), "Folder creation might have failed.");
//         }

//         std::string filename = save_dir + "cloud_frame_" + std::to_string(cloud_save_count) + ".ply";
        
//         // Explicitly use the PLYWriter class to bypass the namespace error
//         pcl::PLYWriter writer;
//         if (writer.write(filename, *laserCloudWorldRGB, true) == 0) {
//             RCLCPP_INFO(this->get_logger(), "Successfully saved colorized cloud: %s", filename.c_str());
//         } else {
//             RCLCPP_ERROR(this->get_logger(), "Failed to save PLY file at %s", filename.c_str());
//         }
//     }
//     cloud_save_count++;
//   }
// }

// void rls::OdomNode::projectLidarToImage(const cv::Mat &img) {
  
//   // 1. Initialize PCL point clouds and transform
//   pcl::PointCloud<PointType>::Ptr voxelized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::PointCloud<PointType>::Ptr deskewed_scan_t_visual = std::make_shared<pcl::PointCloud<PointType>>();
//   pcl::transformPointCloud(*this->deskewed_scan, *deskewed_scan_t_visual, this->T_corr);

//   // 2. Voxel Grid Filtering (1mm resolution)
//   pcl::VoxelGrid<PointType> voxel_filter;
//   voxel_filter.setInputCloud(deskewed_scan_t_visual);
//   voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f);
//   voxel_filter.filter(*voxelized_cloud);
  
//   this->valid_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   this->valid_colorized_points->points.reserve(voxelized_cloud->size());
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudWorldRGB = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
//   laserCloudWorldRGB->reserve(voxelized_cloud->size());

//   // 3. Coordinate Transformation Setup
//   Eigen::Quaternionf q = this->state.q.normalized();
//   this->state.rot = q.toRotationMatrix();
//   Eigen::Matrix3f R_wi = this->state.rot;
//   Eigen::Vector3f P_wi = this->state.p;   
//   Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();
//   Eigen::Vector3f P_cw = -this->Rcl * R_wi.transpose() * P_wi + this->Pcl;

//   if (img.empty()) {
//       RCLCPP_WARN(this->get_logger(), "Image is empty, skipping projection.");
//       return;
//   }

//   // 4. Sharpening Pre-process: Enhance the 2D image before sampling
//   // We use a Laplacian-based sharpening filter to make edges pop
//   cv::Mat sharpened_img;
//   cv::Mat kernel = (cv::Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//   cv::filter2D(img, sharpened_img, img.depth(), kernel);

//   this->projected_pts.clear();
//   this->projected_pts.reserve(voxelized_cloud->size());

//   // 5. Build Z-Buffer for Occlusion Culling
//   cv::Mat depth_buffer(this->camera.height, this->camera.width, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
//   for (int i = 0; i < voxelized_cloud->size(); i++) {
//       Eigen::Vector3f pt(voxelized_cloud->points[i].x, voxelized_cloud->points[i].y, voxelized_cloud->points[i].z);
//       Eigen::Vector3f pc = R_cw * pt + P_cw;
//       if (pc.z() <= 0) continue;
//       Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;
//       int u = static_cast<int>(std::round(image_points(0) / image_points(2)));
//       int v = static_cast<int>(std::round(image_points(1) / image_points(2)));
//       if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height) {
//           if (pc.z() < depth_buffer.at<float>(v, u)) depth_buffer.at<float>(v, u) = pc.z();
//       }
//   }

//   // 6. Project and Sharpen Colors
//   for (int i = 0; i < voxelized_cloud->size(); i++) {
//       Eigen::Vector3f pt(voxelized_cloud->points[i].x, voxelized_cloud->points[i].y, voxelized_cloud->points[i].z);
//       Eigen::Vector3f pc = R_cw * pt + P_cw;
//       if (pc.z() <= 0) continue;

//       Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;
//       float u = image_points(0) / image_points(2);
//       float v = image_points(1) / image_points(2);

//       if (u >= 0 && u < (this->camera.width - 1) && v >= 0 && v < (this->camera.height - 1)) {
//           int u_int = static_cast<int>(std::round(u));
//           int v_int = static_cast<int>(std::round(v));
//           if (pc.z() > depth_buffer.at<float>(v_int, u_int) + 0.3f) continue; 

//           cv::Vec3b raw_color = sharpened_img.at<cv::Vec3b>(v_int, u_int);

//           // --- SHARPENING & CONTRAST MATH ---
//           float b_f = raw_color[0] / 255.0f;
//           float g_f = raw_color[1] / 255.0f;
//           float r_f = raw_color[2] / 255.0f;

//           // 1. Contrast Stretching (Sharpening the color separation)
//           // This pushes colors away from the 'middle gray' to make them pop
//           float contrast = 1.3f; // Adjust to 1.5 for even sharper results
//           r_f = (r_f - 0.5f) * contrast + 0.5f;
//           g_f = (g_f - 0.5f) * contrast + 0.5f;
//           b_f = (b_f - 0.5f) * contrast + 0.5f;

//           // 2. Gamma for Detail Sharpening (High frequency boost)
//           float gamma = 0.85f;
//           r_f = std::pow(std::max(0.0f, r_f), gamma);
//           g_f = std::pow(std::max(0.0f, g_f), gamma);
//           b_f = std::pow(std::max(0.0f, b_f), gamma);

//           // 3. Final Brightness Multiplier
//           float bright_mult = 1.2f;
//           r_f = std::min(1.0f, r_f * bright_mult);
//           g_f = std::min(1.0f, g_f * bright_mult);
//           b_f = std::min(1.0f, b_f * bright_mult);

//           pcl::PointXYZRGB colored_point;
//           colored_point.x = pt(0); colored_point.y = pt(1); colored_point.z = pt(2);
//           colored_point.r = static_cast<uint8_t>(r_f * 255);
//           colored_point.g = static_cast<uint8_t>(g_f * 255);
//           colored_point.b = static_cast<uint8_t>(b_f * 255);
          
//           this->valid_colorized_points->points.push_back(colored_point);
//           laserCloudWorldRGB->push_back(colored_point);
//       }
//   }

//   // 7. ROS Publish and PLY Export
//   if (this->map3d_params_) {
//       sensor_msgs::msg::PointCloud2 rgb_pc;
//       pcl::toROSMsg(*laserCloudWorldRGB, rgb_pc);
//       rgb_pc.header.stamp = this->scan_header_stamp;
//       rgb_pc.header.frame_id = this->odom_frame;
//       this->deskewed_pub->publish(rgb_pc);

//       static int cloud_save_count = 0;
//       if (cloud_save_count % 50 == 0) {
//           std::string save_dir = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data/pointclouds/";
//           int unused = system(("mkdir -p " + save_dir).c_str()); 
//           std::string filename = save_dir + "sharp_cloud_" + std::to_string(cloud_save_count) + ".ply";
//           pcl::io::savePLYFileBinary(filename, *laserCloudWorldRGB);
//       }
//       cloud_save_count++;
//   }
// }

/*
  Projecting LiDAR point to image with Enhanced Colors (Sharpened & Brightened)
*/
void rls::OdomNode::projectLidarToImage(const cv::Mat &img) {
  
  if (img.empty()) {
    RCLCPP_WARN(this->get_logger(), "Image is empty, skipping projection.");
    return;
  }

  // =================================================================
  // --- IMAGE ENHANCEMENT (Sharpening and Brightness) ---
  // =================================================================
  cv::Mat enhanced_img;
  
  // 1. Sharpening Kernel (Unsharp Masking style)
  cv::Mat sharpen_kernel = (cv::Mat_<float>(3,3) << 
                             0, -1,  0,
                            -1,  5, -1,
                             0, -1,  0);
  cv::filter2D(img, enhanced_img, -1, sharpen_kernel);

  // cv::Mat sharpen_kernel = (cv::Mat_<float>(3,3) << 
  //                           -1, -1, -1,
  //                           -1,  9, -1,
  //                           -1, -1, -1);
  // cv::filter2D(img, enhanced_img, -1, sharpen_kernel);

  // 2. Brightness and Contrast Adjustments
  double alpha = 1.4; // Contrast control (1.0 is neutral. 1.2 to 1.5 adds punch)
  int beta = 30;      // Brightness control (0 is neutral. Add 20-40 for darker tunnels)
  enhanced_img.convertTo(enhanced_img, -1, alpha, beta);
  // =================================================================

  // 1. Initialize PCL point clouds
  pcl::PointCloud<PointType>::Ptr voxelized_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_visual = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(*this->deskewed_scan, *deskewed_scan_t_visual, this->T_corr );

  // 2. Voxel Grid Filtering
  pcl::VoxelGrid<PointType> voxel_filter;
  voxel_filter.setInputCloud(deskewed_scan_t_visual);     // Input cloud (transformed)
  voxel_filter.setLeafSize(0.001f, 0.001f, 0.001f);       // Voxel size (adjust as needed)
  voxel_filter.filter(*voxelized_cloud);                  // Output: voxelized_cloud
  
  this->valid_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  this->valid_colorized_points->points.reserve(voxelized_cloud->size());
  this->prev_colorized_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  this->prev_colorized_points->points.reserve(voxelized_cloud->size());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudWorldRGB = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  laserCloudWorldRGB->reserve(voxelized_cloud->size());

  // Extract the state orientation
  Eigen::Quaternionf q = this->state.q.normalized();
  this->state.rot      = q.toRotationMatrix();
  
  // Calculate the transformation between coordinates
  Eigen::Matrix3f R_wi = this->state.rot;                                      // World2imu
  Eigen::Vector3f P_wi = this->state.p;   
  Eigen::Matrix3f R_cw = this->Rcl * R_wi.transpose();                         // Camera2world
  Eigen::Vector3f P_cw = - this->Rcl* R_wi.transpose() * P_wi + this->Pcl;

  this->projected_pts.clear();                                              // Clear the previous points
  this->projected_pts.reserve(voxelized_cloud->size());                     // Pre-allocate memory
  this->prev_projected_pts.reserve(voxelized_cloud->size());

  if (this->visual_initialization) {
    for (int i = 0; i < voxelized_cloud->size(); i++) {
        Eigen::Vector3f pt(voxelized_cloud->points[i].x,
              voxelized_cloud->points[i].y,
              voxelized_cloud->points[i].z);

        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);

        if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
            continue;
        }

        if (u >= 0 && u < (this->camera.width - 1) && v >= 0 && v < (this->camera.height - 1) && image_points(2) > 0) {
          this->projected_pts.emplace_back(u, v);
          int u_floor = static_cast<int>(u);
          int v_floor = static_cast<int>(v);
          float u_frac = u - u_floor;
          float v_frac = v - v_floor;

          // Sample from the ENHANCED image, not the raw one
          cv::Vec3b p00 = enhanced_img.at<cv::Vec3b>(v_floor, u_floor);
          cv::Vec3b p01 = enhanced_img.at<cv::Vec3b>(v_floor, u_floor + 1);
          cv::Vec3b p10 = enhanced_img.at<cv::Vec3b>(v_floor + 1, u_floor);
          cv::Vec3b p11 = enhanced_img.at<cv::Vec3b>(v_floor + 1, u_floor + 1);

          // Interpolate each channel (BGR) separately
          cv::Vec3b color;
          for (int c = 0; c < 3; c++) {
              float channel_val = 
                  (1 - u_frac) * (1 - v_frac) * p00[c] +
                  u_frac * (1 - v_frac) * p01[c] +
                  (1 - u_frac) * v_frac * p10[c] +
                  u_frac * v_frac * p11[c];
              
              // Clamp to ensure interpolation doesn't exceed bounds
              color[c] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, std::round(channel_val))));
          }
          
          pcl::PointXYZRGB colored_point;
          colored_point.x = pt(0);
          colored_point.y = pt(1);
          colored_point.z = pt(2);
          colored_point.r = color[2]; // OpenCV is BGR, PCL is RGB
          colored_point.g = color[1];
          colored_point.b = color[0];
          
          // Add point to rgb point cloud
          this->valid_colorized_points->points.push_back(colored_point);
          laserCloudWorldRGB->push_back(colored_point);
        }
      }
      this->valid_point_flag = true;
      this->cloud_rgb_buffer.push_back(this->valid_colorized_points);
      this->pts_buffer.push_back(this->projected_pts);
  } else {
    for (int i = 0; i < voxelized_cloud->size(); i++) {
        Eigen::Vector3f pt(voxelized_cloud->points[i].x,
              voxelized_cloud->points[i].y,
              voxelized_cloud->points[i].z);

        Eigen::Vector3f   pc =  R_cw * pt + P_cw;
        Eigen::Vector3f image_points = this->camera.intrinsic_matrix * pc;

        // Normalize to get pixel coordinates
        float u = image_points(0) / image_points(2);
        float v = image_points(1) / image_points(2);
        
        if (u < 0 || u >= this->camera.width || v < 0 || v >= this->camera.height || image_points(2) <= 0) {
            continue;
        }

        if (u >= 0 && u < this->camera.width && v >= 0 && v < this->camera.height && image_points(2) > 0) {
          this->prev_projected_pts.emplace_back(u, v);
          
          // Sample from the ENHANCED image
          cv::Vec3b color = enhanced_img.at<cv::Vec3b>(v, u);

          pcl::PointXYZRGB colored_point;
          colored_point.x = pt(0);
          colored_point.y = pt(1);
          colored_point.z = pt(2);
          colored_point.r = color[2];
          colored_point.g = color[1];
          colored_point.b = color[0];
          
          // Add point to rgb point cloud
          this->prev_colorized_points->points.push_back(colored_point);
          laserCloudWorldRGB->push_back(colored_point);
        }
    }
    this->prev_opt_frame = this->prev_frame;
  }

  if (laserCloudWorldRGB->size() > 0) {
    this->visual_initialization = true;
  }

  if (this->valid_colorized_points->size() > 0 && this->projected_pts.size() > 0) {
    this->prev_colorized_points = this->valid_colorized_points;
    this->prev_projected_pts    = this->projected_pts;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_pts_t = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  sensor_msgs::msg::PointCloud2 rgb_pc;
  pcl::toROSMsg(*laserCloudWorldRGB, rgb_pc);
  rgb_pc.header.stamp = this->scan_header_stamp;
  rgb_pc.header.frame_id = this->odom_frame;
  this->deskewed_pub->publish(rgb_pc);
  this->rgb_colorized = true;
}


/*
  PointCloud callback
*/
// void rls::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

//   std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
//   this->main_loop_running = true;
//   lock.unlock();

//   //   double then = this->now().seconds();
//   rclcpp::Clock clock;
//   double then = clock.now().seconds();

//   if (this->first_scan_stamp == 0.) {
//     this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
//   }

//   // rls Initialization procedures (IMU calib, gravity align)
//   if (!this->rls_initialized) {
//     this->initialize();
//   }

//   // Convert incoming scan into format
//   this->getScanFromROS(pc);

//   // Preprocess points
//   this->preprocessPoints();

//   if (!this->first_valid_scan) {
//     return;
//   }

//   if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
//     RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
//     return;
//   }

//   // Compute Metrics
//   this->metrics_thread = std::thread( &rls::OdomNode::computeMetrics, this );
//   this->metrics_thread.detach();

//   // Set Adaptive Parameters
//   if (this->adaptive_params_) {
//     this->setAdaptiveParams();
//   }

//   // Set new frame as input source
//   this->setInputSource();

//   // [NEW] VISUAL ODOMETRY REFINEMENT
//   // -----------------------------------------------------------
//   // 1. Detect features in current frame
//   std::vector<cv::Point2f> curr_features = this->detectFeatures(this->current_intensity_img);
  
//   // 2. Compute motion relative to last frame
//   Eigen::Matrix4f T_vis_delta = this->computeVisualOdometry(
//       this->current_intensity_img, 
//       curr_features, 
//       this->original_scan // Crucial: Must be the ordered cloud!
//   );


//   // // ===================================================================================
//   // // [STEP 2] PREPARE VISUAL DATA (SMOOTHING ONLY)
//   // // ===================================================================================
//   // this->visual_valid_ = false;

//   // if (!T_vis_delta.isIdentity()) {

//   //     // 1. Calculate Raw Motion in Body Frame
//   //     Eigen::Matrix4f T_lidar_motion = T_vis_delta.inverse();
//   //     Eigen::Matrix4f T_ext = this->extrinsics.baselink2lidar_T; 
//   //     Eigen::Matrix4f T_body_motion = T_ext * T_lidar_motion * T_ext.inverse();
      
//   //     Eigen::Vector3f trans_body = T_body_motion.block<3,1>(0,3);
//   //     float dt = this->scan_stamp - this->prev_scan_stamp;
//   //     if (dt <= 0.001) dt = 0.1;

//   //     // 2. Direction Check (Safety)
//   //     float imu_fwd = this->state.v.lin.b.x();
//   //     if (std::abs(imu_fwd) > 0.5 && (imu_fwd * trans_body.x() < 0)) {
//   //         // Skip: Vision is fighting IMU direction
//   //     } 
//   //     else {
//   //         // 3. Exponential Smoothing (Low Pass Filter)
//   //         static Eigen::Vector3f avg_body_vel = Eigen::Vector3f::Zero();
//   //         Eigen::Vector3f curr_vel = trans_body / dt;
          
//   //         if (avg_body_vel.isZero()) avg_body_vel = curr_vel;
          
//   //         // Alpha 0.3 (Trust new data 30%, keep old 70%) - Kills Jitter
//   //         avg_body_vel = 0.7f * avg_body_vel + 0.3f * curr_vel;


//   //         // 4. Store for Solver
//   //         this->visual_body_vel_smoothed_ = avg_body_vel * dt;

//   //         // 5. [FIX] CONFIDENCE-BASED GATING
//   //         // We removed the fixed "speed > 0.2" because it killed your slow motion.
//   //         // Now we use the "Quality" of the match.
          
//   //         float abs_speed = std::abs(avg_body_vel.x());

//   //         // CONDITION A: High Quality (Tunnel)
//   //         // If we have > 50 valid features, the motion is REAL, even if slow.
//   //         bool is_high_quality = (this->visual_inliers_ > 50);

//   //         // CONDITION B: Not "Static Noise"
//   //         // Even with good features, ignore micro-movements (< 1 cm/s) to prevent drift at absolute standstill.
//   //         bool is_moving = (abs_speed > 0.01); 

//   //         if (is_high_quality && is_moving && abs_speed < 1.5) {
              
//   //             this->visual_valid_ = true;
              
//   //             // Debug to confirm it's working
//   //             // RCLCPP_INFO(this->get_logger(), "Fused (Inliers: %d, Speed: %.3f)", this->visual_inliers_, abs_speed);

//   //             // Update Prior to help GICP
//   //             Eigen::Matrix4f T_guess_body = Eigen::Matrix4f::Identity();
//   //             T_guess_body.block<3,1>(0,3) = this->visual_body_vel_smoothed_;
              
//   //             Eigen::Matrix4f T_guess_global = this->T * T_guess_body;
//   //             this->T_prior.block<3,1>(0,3) = T_guess_global.block<3,1>(0,3);

//   //             RCLCPP_INFO(this->get_logger(), "Fused! Visual Speed: %.2f m/s", abs_speed);
//   //         }
//   //     }
//   // }

//   // [NEW] PHASE CORRELATION PIPELINE
//   Eigen::Vector3f visual_motion_body = Eigen::Vector3f::Zero();
  
//   if (this->full_organized_scan_->height > 1) { // Ensure organized
//       visual_motion_body = this->computeDualWallPhaseOdometry(
//           this->current_intensity_img, 
//           this->full_organized_scan_
//       );
//   }

//   // Fusion Logic (Simplified for Phase)
//   this->visual_valid_ = false;
  
//   if (visual_motion_body.norm() > 0.001) { // If successful
      
//       float dt = this->scan_stamp - this->prev_scan_stamp;
//       if (dt <= 0.001) dt = 0.1;
//       float speed = visual_motion_body.x() / dt;

//       // Debug
//       // RCLCPP_INFO(this->get_logger(), "Phase Speed: %.2f m/s", speed);

//       // Gate: 0.1 to 10.0 m/s
//       if (std::abs(speed) > 0.1 && std::abs(speed) < 1.5) {
          
//           // Smooth it
//           static float avg_speed = 0.0f;
//           avg_speed = 0.7f * avg_speed + 0.3f * speed;
          
//           this->visual_body_vel_smoothed_ = Eigen::Vector3f(avg_speed * dt, 0.0, 0.0);
//           this->visual_valid_ = true;
          
//           // Update Prior for GICP
//           Eigen::Matrix4f T_body = Eigen::Matrix4f::Identity();
//           T_body.block<3,1>(0,3) = this->visual_body_vel_smoothed_;
//           this->T_prior = this->T * T_body;
//       }
//   }

//   // 4. Store for next iteration
//   // this->prev_intensity_img_ = this->current_intensity_img.clone();
//   // this->prev_original_scan_ = this->original_scan; // Shared ptr copy is cheap
//   // this->prev_features_      = curr_features;
//   // [HISTORY UPDATE]
//   this->prev_intensity_img_ = this->current_intensity_img.clone();
  
//   // [NEW] Save the Organized cloud
//   this->prev_full_organized_scan_ = this->full_organized_scan_; 
  
//   this->prev_original_scan_ = this->original_scan;
//   this->prev_features_      = curr_features;
//   // -----------------------------------------------------------

//   // Set initial frame as first keyframe
//   if (this->keyframes.size() == 0) {
//     this->initializeInputTarget();
//     this->main_loop_running = false;
//     this->submap_future =
//       std::async( std::launch::async, &rls::OdomNode::buildKeyframesAndSubmap, this, this->state );
//     this->submap_future.wait(); // wait until completion
//     return;
//   }

//   // Get the next pose via IMU + S2M + GEO
//   this->getNextPose();

//   // Modification
//   if (this->img_buffer_.size() > 1) {
//     cv::Mat img = this->img_buffer_.back();
//     this->projectLidarToImage(img);
//   }

//   // Update here:
//   this->curr_lidar_frame.points_corrected = this->deskewed_scan;
//   this->curr_lidar_frame.T_Li_Lk_vec      = this->T;
//   std::vector<int> vec_idx(this->deskewed_scan->size(), 0);
//   this->curr_lidar_frame.vec_idx = vec_idx;

  

//   // Update current keyframe poses and map
//   this->updateKeyframes();

//   // Build keyframe normals and submap if needed (and if we're not already waiting)
//   if (this->new_submap_is_ready) {
//     this->main_loop_running = false;
//     this->submap_future =
//       std::async( std::launch::async, &rls::OdomNode::buildKeyframesAndSubmap, this, this->state );
//   } else {
//     lock.lock();
//     this->main_loop_running = false;
//     lock.unlock();
//     this->submap_build_cv.notify_one();
//   }

//   // Update trajectory
//   this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );

//   // Update time stamps
//   this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
//   this->prev_scan_stamp = this->scan_stamp;
//   this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

//   // Publish stuff to ROS
//   pcl::PointCloud<PointType>::ConstPtr published_cloud;
//   if (this->densemap_filtered_) {
//     published_cloud = this->current_scan;
//   } else {
//     published_cloud = this->deskewed_scan;
//   }
//   this->publish_thread = std::thread( &rls::OdomNode::publishToROS, this, published_cloud, this->T_corr );
//   this->publish_thread.detach();

//   // ===========================================================
//   // [UPDATED] VISUALIZATION (Uses 'curr_features' from Step A)
//   // ===========================================================
//   if (!this->current_intensity_img.empty()) {
      
//       cv::Mat vis_img_gray, vis_img_color;
      
//       // Stretch 5x (Linear Interpolation)
//       float scale_factor = 5.0f;
//       cv::resize(this->current_intensity_img, vis_img_gray, cv::Size(), 1.0, scale_factor, cv::INTER_LINEAR);
      
//       // Convert to Color
//       cv::cvtColor(vis_img_gray, vis_img_color, cv::COLOR_GRAY2BGR);

//       // Draw features (Yellow)
//       // We use 'curr_features' computed at the top of the function
//       for (const auto& pt : curr_features) {
//           cv::Point2f scaled_pt;
//           scaled_pt.x = pt.x; 
//           scaled_pt.y = pt.y * scale_factor; 
//           cv::circle(vis_img_color, scaled_pt, 7, cv::Scalar(0, 255, 255), 2);
//       }

//       // Publish
//       std_msgs::msg::Header header;
//       header.stamp = this->scan_header_stamp;
//       header.frame_id = this->lidar_frame;
      
//       try {
//           sensor_msgs::msg::Image::SharedPtr img_msg = cv_bridge::CvImage(
//               header, "bgr8", vis_img_color).toImageMsg();
//           this->intensity_pub_->publish(*img_msg); 
//       } catch (...) {}
//   }
//   // ===========================================================

//   // Update some statistics
//   //   this->comp_times.push_back(this->now().seconds() - then);
//   this->comp_times.push_back(clock.now().seconds() - then);
//   this->gicp_hasConverged = this->gicp.hasConverged();

//   // Debug statements and publish custom message
//   this->debug_thread = std::thread( &rls::OdomNode::debug, this );
//   this->debug_thread.detach();

//   this->geo.first_opt_done = true;

// }

void rls::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  rclcpp::Clock clock;
  double then = clock.now().seconds();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  if (!this->rls_initialized) {
    this->initialize();
  }

  // 1. Get Scan & Generate Image
  this->getScanFromROS(pc);
  this->preprocessPoints();

  if (!this->first_valid_scan) return;

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    return;
  }

  // Compute Metrics
  this->metrics_thread = std::thread( &rls::OdomNode::computeMetrics, this );
  this->metrics_thread.detach();

  if (this->adaptive_params_) this->setAdaptiveParams();
  this->setInputSource();

  // -----------------------------------------------------------
  // [NEW] VISUAL DEGENERACY FIXER (Phase Correlation)
  // -----------------------------------------------------------
  
  static float last_valid_speed = 0.0f;
  
  Eigen::Vector3f visual_motion_body = Eigen::Vector3f::Zero();
  this->visual_valid_ = false;
  bool measurement_good = false;

  // A. Try to calculate Visual Speed
  if (this->full_organized_scan_->height > 1) { 
      visual_motion_body = this->computeMultiPatchPhaseOdometry(
          this->current_intensity_img, 
          this->full_organized_scan_
      );
      
      if (visual_motion_body.norm() > 0.0001) {
          measurement_good = true;
      }
  }

  float dt = this->scan_stamp - this->prev_scan_stamp;
  if (dt <= 0.001) dt = 0.1;

  // B. Calculate Speed (Measured vs. Coasted)
  float current_speed = 0.0f;
  if (measurement_good) {
      current_speed = visual_motion_body.x() / dt; 
      last_valid_speed = current_speed;
  } else {
      current_speed = last_valid_speed * 0.9f; // Coast if vision lost
  }

  // C. Exponential Smoothing
  static float avg_speed = 0.0f;
  if (avg_speed == 0.0f) avg_speed = current_speed;
  avg_speed = 0.8f * avg_speed + 0.2f * current_speed;

  // D. SAVE SPEED
  this->visual_body_vel_smoothed_ = Eigen::Vector3f(avg_speed * dt, 0.0, 0.0);

  // -----------------------------------------------------------
  // [CRITICAL] DEGENERACY SWITCH (The Logic You Requested)
  // -----------------------------------------------------------
  // 1. GICP fails in Straight Tunnels (Degeneracy).
  // 2. GICP works perfectly in Turns/Corners (Features).
  // Therefore: Enable Vision ONLY when going STRAIGHT.
  
  // Get Yaw Rate (Turning speed)
  float yaw_rate = std::abs(this->state.v.ang.b.z());

  // Condition 1: Are we moving? (Ignore static noise > 0.05 m/s)
  bool is_moving = std::abs(avg_speed) < 1.5;

  // Condition 2: Are we going STRAIGHT? (Yaw rate < 0.2 rad/s approx 11 deg/s)
  bool is_straight = yaw_rate < 0.05;

  if (is_moving && is_straight) {
      // ROBOT IS IN A TUNNEL/HALLWAY -> ENABLE VISUAL FIX
      this->visual_valid_ = true;
      
      Eigen::Matrix4f T_body = Eigen::Matrix4f::Identity();
      T_body.block<3,1>(0,3) = this->visual_body_vel_smoothed_;
      this->T_prior = this->T * T_body;
  } 
  else {
      // ROBOT IS TURNING OR STOPPED -> TRUST GICP (Disable Vision)
      this->visual_valid_ = false;
  }
  // -----------------------------------------------------------

  if (this->debug_phase_pose_pub_->get_subscription_count() > 0) {
      Eigen::Vector3f global_step = this->state.q * this->visual_body_vel_smoothed_;
      this->phase_accumulated_pos_ += global_step;
      
      geometry_msgs::msg::PoseStamped phase_pose;
      phase_pose.header.stamp = this->scan_header_stamp;
      phase_pose.header.frame_id = this->odom_frame; 
      phase_pose.pose.position.x = this->phase_accumulated_pos_.x();
      phase_pose.pose.position.y = this->phase_accumulated_pos_.y();
      phase_pose.pose.position.z = this->phase_accumulated_pos_.z();
      phase_pose.pose.orientation.w = this->state.q.w();
      phase_pose.pose.orientation.x = this->state.q.x();
      phase_pose.pose.orientation.y = this->state.q.y();
      phase_pose.pose.orientation.z = this->state.q.z();
      this->debug_phase_pose_pub_->publish(phase_pose);
  }

  this->prev_intensity_img_ = this->current_intensity_img.clone();

  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future = std::async( std::launch::async, &rls::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); 
    return;
  }

  this->getNextPose();

  if (this->img_buffer_.size() > 1) {
    cv::Mat img = this->img_buffer_.back();
    this->projectLidarToImage(img);
  }
   
  this->updateKeyframes();

  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future = std::async( std::launch::async, &rls::OdomNode::buildKeyframesAndSubmap, this, this->state );
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) published_cloud = this->current_scan;
  else published_cloud = this->deskewed_scan;
   
  this->publish_thread = std::thread( &rls::OdomNode::publishToROS, this, published_cloud, this->T_corr );
  this->publish_thread.detach();

  this->comp_times.push_back(clock.now().seconds() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();
  this->debug_thread = std::thread( &rls::OdomNode::debug, this );
  this->debug_thread.detach();
  this->geo.first_opt_done = true;
}

// double rls::OdomNode::getPhaseShift(const cv::Mat& img1, const cv::Mat& img2) {
//     if (img1.empty() || img2.empty()) return 0.0;

//     cv::Mat i1, i2;
//     img1.convertTo(i1, CV_32F);
//     img2.convertTo(i2, CV_32F);

//     // Apply Hanning Window to reduce edge noise in FFT
//     cv::Mat hanningWin;
//     cv::createHanningWindow(hanningWin, i1.size(), CV_32F);
//     i1 = i1.mul(hanningWin);
//     i2 = i2.mul(hanningWin);

//     // Phase Correlate
//     // Returns the (x,y) shift required to align img1 to img2
//     cv::Point2d shift = cv::phaseCorrelate(i1, i2);
    
//     // We only care about horizontal shift (Azimuth change) for forward motion
//     return shift.x; 
// }

// Returns pair: {Shift_X, Response_Confidence}
std::pair<double, double> rls::OdomNode::getPhaseShift(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return {0.0, 0.0};

    cv::Mat i1, i2;
    img1.convertTo(i1, CV_32F);
    img2.convertTo(i2, CV_32F);

    // Hanning Window reduces edge noise
    cv::Mat hanningWin;
    cv::createHanningWindow(hanningWin, i1.size(), CV_32F);
    i1 = i1.mul(hanningWin);
    i2 = i2.mul(hanningWin);

    // Phase Correlate with Response
    double response = 0.0;
    cv::Point2d shift = cv::phaseCorrelate(i1, i2, hanningWin, &response);
    
    if (std::isnan(shift.x) || std::isnan(response)) return {0.0, 0.0};

    // Return horizontal shift and confidence score (PSR)
    return {shift.x, response}; 
}

// Eigen::Vector3f rls::OdomNode::computeMultiPatchPhaseOdometry(
//     const cv::Mat& curr_img, 
//     const pcl::PointCloud<PointType>::ConstPtr& cloud) 
// {
//     if (this->prev_intensity_img_.empty()) return Eigen::Vector3f::Zero();

//     int w = curr_img.cols;
//     int h = curr_img.rows;
//     double ang_res = (2.0 * M_PI) / w; 

//     // 1. GENERATE CANDIDATE ROIS
//     std::vector<cv::Rect> candidates;
//     int roi_w = w / 8;   
//     int roi_h = h * 0.4; 
    
//     for (int c = 0; c < w * 0.35; c += roi_w/2) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6)); // Left
//     for (int c = w * 0.65; c < w - roi_w; c += roi_w/2) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6)); // Right

//     // 2. PROCESS
//     std::vector<float> signed_speeds; 
//     std::vector<cv::Rect> debug_good_rois;

//     for (const auto& roi : candidates) {
//         if (roi.x + roi.width > w) continue;
//         cv::Mat patch = curr_img(roi);
//         if (cv::sum(patch)[0] < 5000) continue; 

//         double shift = getPhaseShift(this->prev_intensity_img_(roi), patch);
//         if (std::abs(shift) > 40.0) continue; 

//         double depth_sum = 0.0;
//         int count = 0;
//         for (int r = roi.y; r < roi.y + roi.height; r+=8) {
//             for (int c = roi.x; c < roi.x + roi.width; c+=8) {
//                 int idx = r * w + c;
//                 if (idx < cloud->size()) {
//                     float d = std::hypot(cloud->points[idx].x, cloud->points[idx].y);
//                     if (std::isfinite(d) && d > 1.0) { depth_sum += d; count++; }
//                 }
//             }
//         }

//         if (count < 50) continue; 
//         double depth = depth_sum / count;
        
//         // Robot Forward = Texture Backward => Speed = 1.0 * Shift (Positive correlation setup)
//         double speed_contribution = 1.0 * shift * ang_res * depth;
        
//         signed_speeds.push_back(speed_contribution);
//         debug_good_rois.push_back(roi);
//     }

//     // 3. FUSION
//     float final_speed = 0.0f;
//     bool success = !signed_speeds.empty(); // [FIX] This defines the variable!

//     if (success) {
//         std::sort(signed_speeds.begin(), signed_speeds.end());
//         double median_speed = signed_speeds[signed_speeds.size() / 2];
//         double speed_sum = 0.0;
//         int inlier_count = 0;
//         for (float s : signed_speeds) {
//             if (std::abs(s - median_speed) < 0.5) { 
//                 speed_sum += s;
//                 inlier_count++;
//             }
//         }
//         final_speed = (inlier_count > 0) ? (speed_sum / inlier_count) : median_speed;
//     }

//     // 4. VISUALIZATION (Scaled Up 5x)
//     if (this->debug_phase_img_pub_->get_subscription_count() > 0) {
        
//         cv::Mat debug_img_small;
//         if (curr_img.type() == CV_8UC1) cv::cvtColor(curr_img, debug_img_small, cv::COLOR_GRAY2BGR);
//         else curr_img.copyTo(debug_img_small);

//         // Scale Up 5x Vertically
//         float scale_y = 5.0f;
//         cv::Mat debug_img;
//         cv::resize(debug_img_small, debug_img, cv::Size(), 1.0, scale_y, cv::INTER_LINEAR);

//         // Draw Rectangles (Scaled)
//         for (const auto& roi : debug_good_rois) {
//             cv::Rect scaled_roi(roi.x, roi.y * scale_y, roi.width, roi.height * scale_y);
//             cv::rectangle(debug_img, scaled_roi, cv::Scalar(0, 255, 0), 2);
//         }
        
//         // Draw Text
//         std::string txt;
//         cv::Scalar color;
//         if (success) {
//             color = (final_speed > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
//             txt = "Speed: " + std::to_string(final_speed).substr(0,5) + " m/s";
//         } else {
//             color = cv::Scalar(0, 0, 255); // Red
//             txt = "NO LOCK";
//         }

//         // Draw Text (Larger and Centered)
//         int text_y = debug_img.rows - 70; 
//         cv::rectangle(debug_img, cv::Point(20, text_y - 60), cv::Point(600, text_y + 20), cv::Scalar(0,0,0), -1);
//         cv::putText(debug_img, txt, cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

//         // Publish
//         std_msgs::msg::Header header;
//         header.stamp = this->scan_header_stamp;
//         header.frame_id = this->lidar_frame;
//         this->debug_phase_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", debug_img).toImageMsg());
//     }

//     if (!success) return Eigen::Vector3f::Zero();

//     return Eigen::Vector3f(final_speed, 0.0, 0.0);
// }

// Eigen::Vector3f rls::OdomNode::computeMultiPatchPhaseOdometry(
//     const cv::Mat& curr_img, 
//     const pcl::PointCloud<PointType>::ConstPtr& cloud) 
// {
//     if (this->prev_intensity_img_.empty()) return Eigen::Vector3f::Zero();

//     int w = curr_img.cols;
//     int h = curr_img.rows;
//     double ang_res = (2.0 * M_PI) / w; 

//     // 1. GENERATE CANDIDATE ROIS
//     std::vector<cv::Rect> candidates;
//     int roi_w = w / 8;   
//     int roi_h = h * 0.4; 
    
//     for (int c = 0; c < w * 0.35; c += roi_w/2) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6)); // Left
//     for (int c = w * 0.65; c < w - roi_w; c += roi_w/2) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6)); // Right

//     // 2. PROCESS
//     std::vector<float> signed_speeds; 
//     std::vector<cv::Rect> debug_good_rois;

//     for (const auto& roi : candidates) {
//         if (roi.x + roi.width > w) continue;
//         cv::Mat patch = curr_img(roi);
//         if (cv::sum(patch)[0] < 5000) continue; 

//         double shift = getPhaseShift(this->prev_intensity_img_(roi), patch);
//         if (std::abs(shift) > 40.0) continue; 

//         double depth_sum = 0.0;
//         int count = 0;
//         for (int r = roi.y; r < roi.y + roi.height; r+=8) {
//             for (int c = roi.x; c < roi.x + roi.width; c+=8) {
//                 int idx = r * w + c;
//                 if (idx < cloud->size()) {
//                     float d = std::hypot(cloud->points[idx].x, cloud->points[idx].y);
//                     if (std::isfinite(d) && d > 1.0) { depth_sum += d; count++; }
//                 }
//             }
//         }

//         if (count < 50) continue; 
//         double depth = depth_sum / count;
        
//         // Robot Forward = Texture Backward => Speed = 1.0 * Shift
//         double speed_contribution = 1.0 * shift * ang_res * depth;
        
//         signed_speeds.push_back(speed_contribution);
//         debug_good_rois.push_back(roi);
//     }

//     // 3. FUSION
//     float final_speed = 0.0f;
//     bool success = !signed_speeds.empty();

//     if (success) {
//         std::sort(signed_speeds.begin(), signed_speeds.end());
//         double median_speed = signed_speeds[signed_speeds.size() / 2];
//         double speed_sum = 0.0;
//         int inlier_count = 0;
//         for (float s : signed_speeds) {
//             if (std::abs(s - median_speed) < 0.5) { 
//                 speed_sum += s;
//                 inlier_count++;
//             }
//         }
//         final_speed = (inlier_count > 0) ? (speed_sum / inlier_count) : median_speed;
//     }

//     // -------------------------------------------------------------------------
//     // 4. "FANCY" PYRAMID VISUALIZATION
//     // -------------------------------------------------------------------------
//     if (this->debug_phase_img_pub_->get_subscription_count() > 0) {
        
//         // A. Prepare Base Images
//         cv::Mat raw_img;
//         if (curr_img.type() == CV_8UC1) cv::cvtColor(curr_img, raw_img, cv::COLOR_GRAY2BGR);
//         else curr_img.copyTo(raw_img);

//         // B. Create Heatmap (Layer 2: Texture Energy)
//         cv::Mat heatmap = cv::Mat::zeros(raw_img.size(), CV_8UC3);
        
//         for (int c = 0; c < w; c+=4) { // Scan columns
//             cv::Mat roi = curr_img(cv::Rect(c, h*0.2, 4, h*0.6)); // Vertical strip
//             double energy = cv::sum(roi)[0];
            
//             // Normalize energy
//             double norm_e = std::min(energy / 50000.0, 1.0);
            
//             cv::Vec3b color;
//             if (norm_e < 0.1) color = cv::Vec3b(50, 0, 0); // Black/Blueish
//             else {
//                 // [FIXED] Use static_cast to clear warnings
//                 int r = static_cast<int>(255 * norm_e);
//                 int g = static_cast<int>(255 * (1.0 - std::abs(norm_e - 0.5)*2.0));
//                 int b = static_cast<int>(255 * (1.0 - norm_e));
//                 color = cv::Vec3b(b, g, r);
//             }
//             cv::rectangle(heatmap, cv::Rect(c, 0, 4, h), color, -1);
//         }

//         // C. Create Flow Field (Layer 3: Motion Vectors)
//         cv::Mat flow_map = cv::Mat::zeros(raw_img.size(), CV_8UC3);
//         flow_map.setTo(cv::Scalar(20, 20, 20)); // Dark Gray background

//         for (size_t i = 0; i < candidates.size(); i++) {
//             cv::Rect roi = candidates[i];
            
//             // Safety check for index bounds
//             if (i >= signed_speeds.size()) break;
            
//             float speed = signed_speeds[i]; 
            
//             cv::Point center(roi.x + roi.width/2, roi.y + roi.height/2);
//             cv::Point arrow_end = center + cv::Point(speed * 500.0, 0); 
            
//             cv::Scalar arrow_color;
//             if (std::abs(speed - final_speed) < 0.2) arrow_color = cv::Scalar(0, 255, 0); // Green
//             else arrow_color = cv::Scalar(0, 0, 255); // Red (Outlier)

//             cv::arrowedLine(flow_map, center, arrow_end, arrow_color, 2);
//             cv::rectangle(flow_map, roi, cv::Scalar(50, 50, 50), 1); 
//         }

//         // D. Stack Them Vertically
//         cv::Mat canvas;
//         std::vector<cv::Mat> layers = {raw_img, heatmap, flow_map};
//         cv::vconcat(layers, canvas);

//         // [Scale Up] Stretch 3x vertically
//         cv::resize(canvas, canvas, cv::Size(), 1.0, 3.0, cv::INTER_NEAREST);

//         // E. Overlay Text
//         std::string status_txt;
//         cv::Scalar status_color;
//         if (success) {
//             status_color = (final_speed > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
//             status_txt = "SPEED: " + std::to_string(final_speed).substr(0,5) + " m/s";
//         } else {
//             status_color = cv::Scalar(0, 0, 255);
//             status_txt = "NO LOCK";
//         }

//         // Labels
//         int h_scaled = h * 3.0; 
//         cv::putText(canvas, "1. RAW INTENSITY", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
//         cv::putText(canvas, "2. TEXTURE ENERGY", cv::Point(10, h_scaled + 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
//         cv::putText(canvas, "3. PHASE FLOW FIELD", cv::Point(10, h_scaled*2 + 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);

//         // Main Speed
//         int bottom_y = canvas.rows - 20;
//         cv::rectangle(canvas, cv::Point(0, bottom_y - 60), cv::Point(600, bottom_y + 20), cv::Scalar(0,0,0), -1);
//         cv::putText(canvas, status_txt, cv::Point(10, bottom_y), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);

//         // Publish
//         std_msgs::msg::Header header;
//         header.stamp = this->scan_header_stamp;
//         header.frame_id = this->lidar_frame;
//         this->debug_phase_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg());
//     }

//     if (!success) return Eigen::Vector3f::Zero();

//     return Eigen::Vector3f(final_speed, 0.0, 0.0);
// }

// Eigen::Vector3f rls::OdomNode::computeMultiPatchPhaseOdometry(
//     const cv::Mat& curr_img, 
//     const pcl::PointCloud<PointType>::ConstPtr& cloud) 
// {
//     if (this->prev_intensity_img_.empty()) return Eigen::Vector3f::Zero();

//     int w = curr_img.cols;
//     int h = curr_img.rows;
//     double ang_res = (2.0 * M_PI) / w; 

//     // -------------------------------------------------------------------------
//     // 1. GENERATE CANDIDATE ROIS (Strictly Walls Only)
//     // -------------------------------------------------------------------------
//     std::vector<cv::Rect> candidates;
//     int roi_w = w / 8;   
//     int roi_h = h * 0.4; 
//     int step = roi_w / 2; // Overlap for robustness
    
//     // Scan Left Wall (0% to 35%)
//     for (int c = 0; c < w * 0.35; c += step) {
//          candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));
//     }
//     // Scan Right Wall (65% to 100% - roi_w)
//     for (int c = w * 0.65; c < w - roi_w; c += step) {
//          candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));
//     }

//     // -------------------------------------------------------------------------
//     // 2. PROCESS & COLLECT DATA
//     // -------------------------------------------------------------------------
//     std::vector<float> signed_speeds; 
//     std::vector<cv::Rect> debug_good_rois;
    
//     // Visualization Buffers (mapped to image width)
//     std::vector<double> vis_energy(w, 0.0);
//     std::vector<double> vis_confidence(w, 0.0);

//     for (const auto& roi : candidates) {
//         if (roi.x + roi.width > w) continue; 

//         cv::Mat patch = curr_img(roi);
        
//         // 1. Energy
//         double energy = cv::sum(patch)[0];
//         // Fill buffer for visualization
//         for(int k=roi.x; k<roi.x+roi.width && k<w; k+=4) vis_energy[k] = energy;

//         if (energy < 5000) continue; 

//         // 2. Phase Shift & Confidence
//         std::pair<double, double> result = getPhaseShift(this->prev_intensity_img_(roi), patch);
//         double shift = result.first;
//         double response = result.second;
        
//         // Fill buffer for visualization
//         for(int k=roi.x; k<roi.x+roi.width && k<w; k+=4) vis_confidence[k] = response;

//         // Filters
//         if (response < 0.02) continue; 
//         if (std::abs(shift) > 40.0) continue; 

//         // 3. Depth
//         double depth_sum = 0.0;
//         int count = 0;
//         for (int r = roi.y; r < roi.y + roi.height; r+=8) {
//             for (int c_pt = roi.x; c_pt < roi.x + roi.width; c_pt+=8) {
//                 int idx = r * w + c_pt;
//                 if (idx < cloud->size()) {
//                     float d = std::hypot(cloud->points[idx].x, cloud->points[idx].y);
//                     if (std::isfinite(d) && d > 1.0) { depth_sum += d; count++; }
//                 }
//             }
//         }

//         if (count < 50) continue; 
//         double depth = depth_sum / count;
        
//         double speed_contribution = 1.0 * shift * ang_res * depth;
//         signed_speeds.push_back(speed_contribution);
//         debug_good_rois.push_back(roi);
//     }

//     // -------------------------------------------------------------------------
//     // 3. FUSION
//     // -------------------------------------------------------------------------
//     float final_speed = 0.0f;
//     bool success = !signed_speeds.empty();

//     if (success) {
//         std::sort(signed_speeds.begin(), signed_speeds.end());
//         double median_speed = signed_speeds[signed_speeds.size() / 2];
//         double speed_sum = 0.0;
//         int inlier_count = 0;
//         for (float s : signed_speeds) {
//             if (std::abs(s - median_speed) < 0.5) { 
//                 speed_sum += s;
//                 inlier_count++;
//             }
//         }
//         final_speed = (inlier_count > 0) ? (speed_sum / inlier_count) : median_speed;
//     }

//     // -------------------------------------------------------------------------
//     // 4. VISUALIZATION
//     // -------------------------------------------------------------------------
//     if (this->debug_phase_img_pub_->get_subscription_count() > 0) {
        
//         // --- Layer 1: RAW ---
//         cv::Mat raw_img;
//         if (curr_img.type() == CV_8UC1) cv::cvtColor(curr_img, raw_img, cv::COLOR_GRAY2BGR);
//         else curr_img.copyTo(raw_img);
//         for (const auto& roi : debug_good_rois) cv::rectangle(raw_img, roi, cv::Scalar(0, 255, 0), 2);

//         // --- Layer 2: TEXTURE ENERGY (Fixed Scale) ---
//         // [FIX] Use fixed scaling instead of auto-normalization
//         cv::Mat energy_gray(1, w, CV_8UC1);
//         const double FIXED_ENERGY_SCALE = 50000.0; // Adjust based on lidar brightness

//         for (int i = 0; i < w; ++i) {
//             // Scale by fixed amount and clamp to 0-1 range
//             double norm_val = std::min(vis_energy[i] / FIXED_ENERGY_SCALE, 1.0);
//             // Map to 0-255 grayscale
//             energy_gray.at<uchar>(0, i) = static_cast<uchar>(norm_val * 255);
//         }
//         cv::Mat energy_color;
//         cv::applyColorMap(energy_gray, energy_color, cv::COLORMAP_TURBO);
//         cv::resize(energy_color, energy_color, raw_img.size(), 0, 0, cv::INTER_LINEAR);

//         // --- Layer 3: PHASE CONFIDENCE (Auto-Normalized) ---
//         // Keep this normalized to show relative strength
//         cv::Mat conf_strip(1, w, CV_64F, vis_confidence.data());
//         cv::Mat conf_norm, conf_color;
//         cv::normalize(conf_strip, conf_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
//         cv::applyColorMap(conf_norm, conf_color, cv::COLORMAP_VIRIDIS);
//         cv::resize(conf_color, conf_color, raw_img.size(), 0, 0, cv::INTER_LINEAR);

//         // --- STACK ---
//         cv::Mat canvas;
//         std::vector<cv::Mat> layers = {raw_img, energy_color, conf_color};
//         cv::vconcat(layers, canvas);
//         cv::resize(canvas, canvas, cv::Size(), 1.0, 3.0, cv::INTER_NEAREST); 

//         // Text & Overlay
//         std::string status_txt = success ? "SPEED: " + std::to_string(final_speed).substr(0,5) + " m/s" : "NO LOCK";
//         cv::Scalar status_color = success ? ((final_speed>0)?cv::Scalar(0,255,0):cv::Scalar(0,0,255)) : cv::Scalar(0,0,255);
//         int h_scaled = h * 3.0; 
        
//         auto drawLabel = [&](const std::string& text, cv::Point pt) {
//             cv::putText(canvas, text, pt + cv::Point(2,2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
//             cv::putText(canvas, text, pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
//         };
//         drawLabel("1. RAW INTENSITY (Green = Active ROI)", cv::Point(10, 30));
//         drawLabel("2. TEXTURE ENERGY (Fixed Scale)", cv::Point(10, h_scaled + 30));
//         drawLabel("3. PHASE CONFIDENCE (Normalized)", cv::Point(10, h_scaled*2 + 30));

//         int bottom_y = canvas.rows - 20;
//         cv::rectangle(canvas, cv::Point(0, bottom_y - 60), cv::Point(600, bottom_y + 20), cv::Scalar(0,0,0), -1);
//         cv::putText(canvas, status_txt, cv::Point(20, bottom_y), cv::FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4);

//         std_msgs::msg::Header header;
//         header.stamp = this->scan_header_stamp;
//         header.frame_id = this->lidar_frame;
//         this->debug_phase_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg());
//     }

//     if (!success) return Eigen::Vector3f::Zero();

//     return Eigen::Vector3f(final_speed, 0.0, 0.0);
// }

// Eigen::Vector3f rls::OdomNode::computeMultiPatchPhaseOdometry(
//     const cv::Mat& curr_img, 
//     const pcl::PointCloud<PointType>::ConstPtr& cloud) 
// {
//     if (this->prev_intensity_img_.empty()) return Eigen::Vector3f::Zero();

//     int w = curr_img.cols;
//     int h = curr_img.rows;
//     double ang_res = (2.0 * M_PI) / w; 

//     // -------------------------------------------------------------------------
//     // 1. GENERATE CANDIDATE ROIS (Strictly Walls Only)
//     // -------------------------------------------------------------------------
//     std::vector<cv::Rect> candidates;
//     int roi_w = w / 8;   
//     int roi_h = h * 0.4; 
//     int step = roi_w / 2; // Overlap for robustness
    
//     // Scan Left Wall (0% to 35%)
//     for (int c = 0; c < w * 0.35; c += step) {
//          candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));
//     }
//     // Scan Right Wall (65% to 100% - roi_w)
//     for (int c = w * 0.65; c < w - roi_w; c += step) {
//          candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));
//     }

//     // -------------------------------------------------------------------------
//     // 2. PROCESS & COLLECT DATA
//     // -------------------------------------------------------------------------
//     std::vector<float> signed_speeds; 
//     std::vector<cv::Rect> debug_good_rois;
    
//     // Visualization Buffer for Phase Confidence (Layer 3)
//     // We initialize with 0.0 so the center (which we skip) appears dark.
//     std::vector<double> vis_confidence(w, 0.0);

//     for (const auto& roi : candidates) {
//         if (roi.x + roi.width > w) continue; 

//         cv::Mat patch = curr_img(roi);
//         double energy = cv::sum(patch)[0];

//         if (energy < 5000) continue; 

//         // 2. Phase Shift & Confidence
//         std::pair<double, double> result = getPhaseShift(this->prev_intensity_img_(roi), patch);
//         double shift = result.first;
//         double response = result.second;
        
//         // Fill buffer for Layer 3 Visualization
//         for(int k=roi.x; k<roi.x+roi.width && k<w; k+=4) vis_confidence[k] = response;

//         // Filters
//         if (response < 0.02) continue; 
//         if (std::abs(shift) > 40.0) continue; 

//         // 3. Depth
//         double depth_sum = 0.0;
//         int count = 0;
//         for (int r = roi.y; r < roi.y + roi.height; r+=8) {
//             for (int c_pt = roi.x; c_pt < roi.x + roi.width; c_pt+=8) {
//                 int idx = r * w + c_pt;
//                 if (idx < cloud->size()) {
//                     float d = std::hypot(cloud->points[idx].x, cloud->points[idx].y);
//                     if (std::isfinite(d) && d > 1.0) { depth_sum += d; count++; }
//                 }
//             }
//         }

//         if (count < 50) continue; 
//         double depth = depth_sum / count;
        
//         double speed_contribution = 1.0 * shift * ang_res * depth;
//         signed_speeds.push_back(speed_contribution);
//         debug_good_rois.push_back(roi);
//     }

//     // -------------------------------------------------------------------------
//     // 3. FUSION
//     // -------------------------------------------------------------------------
//     float final_speed = 0.0f;
//     bool success = !signed_speeds.empty();

//     if (success) {
//         std::sort(signed_speeds.begin(), signed_speeds.end());
//         double median_speed = signed_speeds[signed_speeds.size() / 2];
//         double speed_sum = 0.0;
//         int inlier_count = 0;
//         for (float s : signed_speeds) {
//             if (std::abs(s - median_speed) < 0.5) { 
//                 speed_sum += s;
//                 inlier_count++;
//             }
//         }
//         final_speed = (inlier_count > 0) ? (speed_sum / inlier_count) : median_speed;
//     }

//     // -------------------------------------------------------------------------
//     // 4. VISUALIZATION (Manual Texture Gradient + Auto Phase Map)
//     // -------------------------------------------------------------------------
//     if (this->debug_phase_img_pub_->get_subscription_count() > 0) {
        
//         // --- Layer 1: RAW ---
//         cv::Mat raw_img;
//         if (curr_img.type() == CV_8UC1) cv::cvtColor(curr_img, raw_img, cv::COLOR_GRAY2BGR);
//         else curr_img.copyTo(raw_img);
//         for (const auto& roi : debug_good_rois) cv::rectangle(raw_img, roi, cv::Scalar(0, 255, 0), 2);

//         // --- Layer 2: TEXTURE ENERGY (Manual Gradient Style) ---
//         // This generates the visual style you preferred
//         cv::Mat heatmap = cv::Mat::zeros(raw_img.size(), CV_8UC3);
        
//         for (int c = 0; c < w; c+=4) { // Scan columns
//             cv::Mat roi = curr_img(cv::Rect(c, h*0.2, 4, h*0.6)); // Vertical strip
//             double energy = cv::sum(roi)[0];
            
//             // Fixed Scale Normalization (50,000)
//             double norm_e = std::min(energy / 50000.0, 1.0);
            
//             cv::Vec3b color;
//             if (norm_e < 0.1) color = cv::Vec3b(50, 0, 0); // Black/Blueish
//             else {
//                 int r = static_cast<int>(255 * norm_e);
//                 int g = static_cast<int>(255 * (1.0 - std::abs(norm_e - 0.5)*2.0));
//                 int b = static_cast<int>(255 * (1.0 - norm_e));
//                 color = cv::Vec3b(b, g, r);
//             }
//             cv::rectangle(heatmap, cv::Rect(c, 0, 4, h), color, -1);
//         }

//         // --- Layer 3: PHASE CONFIDENCE (Auto-Normalized Viridis) ---
//         // This remains auto-scaled so you can see "lock quality" clearly
//         cv::Mat conf_strip(1, w, CV_64F, vis_confidence.data());
//         cv::Mat conf_norm, conf_color;
//         cv::normalize(conf_strip, conf_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
//         cv::applyColorMap(conf_norm, conf_color, cv::COLORMAP_VIRIDIS);
//         cv::resize(conf_color, conf_color, raw_img.size(), 0, 0, cv::INTER_LINEAR);

//         // --- STACK ---
//         cv::Mat canvas;
//         std::vector<cv::Mat> layers = {raw_img, heatmap, conf_color};
//         cv::vconcat(layers, canvas);
//         cv::resize(canvas, canvas, cv::Size(), 1.0, 3.0, cv::INTER_NEAREST); 

//         // Text & Overlay
//         std::string status_txt = success ? "SPEED: " + std::to_string(final_speed).substr(0,5) + " m/s" : "NO LOCK";
//         cv::Scalar status_color = success ? ((final_speed>0)?cv::Scalar(0,255,0):cv::Scalar(0,0,255)) : cv::Scalar(0,0,255);
//         int h_scaled = h * 3.0; 
        
//         auto drawLabel = [&](const std::string& text, cv::Point pt) {
//             cv::putText(canvas, text, pt + cv::Point(2,2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
//             cv::putText(canvas, text, pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
//         };
//         drawLabel("1. RAW INTENSITY (Green = Active ROI)", cv::Point(10, 30));
//         drawLabel("2. TEXTURE ENERGY (Fixed Scale)", cv::Point(10, h_scaled + 30));
//         drawLabel("3. PHASE CONFIDENCE (Normalized)", cv::Point(10, h_scaled*2 + 30));

//         int bottom_y = canvas.rows - 20;
//         cv::rectangle(canvas, cv::Point(0, bottom_y - 60), cv::Point(600, bottom_y + 20), cv::Scalar(0,0,0), -1);
//         cv::putText(canvas, status_txt, cv::Point(20, bottom_y), cv::FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4);

//         std_msgs::msg::Header header;
//         header.stamp = this->scan_header_stamp;
//         header.frame_id = this->lidar_frame;
//         this->debug_phase_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg());
//     }

//     if (!success) return Eigen::Vector3f::Zero();

//     return Eigen::Vector3f(final_speed, 0.0, 0.0);
// }

Eigen::Vector3f rls::OdomNode::computeMultiPatchPhaseOdometry(
    const cv::Mat& curr_img, 
    const pcl::PointCloud<PointType>::ConstPtr& cloud) 
{
    if (this->prev_intensity_img_.empty()) return Eigen::Vector3f::Zero();

    int w = curr_img.cols;
    int h = curr_img.rows;
    double ang_res = (2.0 * M_PI) / w; 

    // -------------------------------------------------------------------------
    // 0. IMU COMPENSATION (Critical for Turning/Curves)
    // -------------------------------------------------------------------------
    // Calculate how many pixels the image shifted purely due to rotation
    float dt = this->scan_stamp - this->prev_scan_stamp;
    if (dt <= 0.0001) dt = 0.1;

    // Yaw Rate from IMU (rad/s)
    float yaw_rate = this->state.v.ang.b.z(); 
    
    // Rotation Shift in Pixels = (Radians) / (Radians per Pixel)
    // Note: Sign depends on sensor direction. Usually Turn Left (+) -> Image Shifts Right (+)
    // You might need to flip this sign to (-yaw_rate) if turning corrects the wrong way.
    double rotation_shift_px = (yaw_rate * dt) / ang_res;

    // -------------------------------------------------------------------------
    // 1. GENERATE CANDIDATE ROIS
    // -------------------------------------------------------------------------
    std::vector<cv::Rect> candidates;
    int roi_w = w / 8;   
    int roi_h = h * 0.4; 
    int step = roi_w / 2; 
    
    // Left Wall
    for (int c = 0; c < w * 0.35; c += step) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));
    // Right Wall
    for (int c = w * 0.65; c < w - roi_w; c += step) candidates.push_back(cv::Rect(c, h*0.2, roi_w, h*0.6));

    // -------------------------------------------------------------------------
    // 2. PROCESS & COLLECT DATA
    // -------------------------------------------------------------------------
    std::vector<float> signed_speeds; 
    std::vector<cv::Rect> debug_good_rois;
    
    std::vector<double> vis_energy(w, 0.0);
    std::vector<double> vis_confidence(w, 0.0);

    for (const auto& roi : candidates) {
        if (roi.x + roi.width > w) continue; 

        cv::Mat patch = curr_img(roi);
        double energy = cv::sum(patch)[0];
        
        for(int k=roi.x; k<roi.x+roi.width && k<w; k+=4) vis_energy[k] = energy;

        if (energy < 5000) continue; 

        // Get Raw Phase Shift (Total Shift)
        std::pair<double, double> result = getPhaseShift(this->prev_intensity_img_(roi), patch);
        double total_shift = result.first;
        double response = result.second;
        
        for(int k=roi.x; k<roi.x+roi.width && k<w; k+=4) vis_confidence[k] = response;

        if (response < 0.02) continue; 
        if (std::abs(total_shift) > 40.0) continue; 

        // [FIX] SUBTRACT ROTATION
        // We remove the shift caused by turning, leaving only the shift caused by moving forward.
        double translation_shift = total_shift - rotation_shift_px;

        // Depth
        double depth_sum = 0.0;
        int count = 0;
        for (int r = roi.y; r < roi.y + roi.height; r+=8) {
            for (int c_pt = roi.x; c_pt < roi.x + roi.width; c_pt+=8) {
                int idx = r * w + c_pt;
                if (idx < cloud->size()) {
                    float d = std::hypot(cloud->points[idx].x, cloud->points[idx].y);
                    if (std::isfinite(d) && d > 1.0) { depth_sum += d; count++; }
                }
            }
        }

        if (count < 50) continue; 
        double depth = depth_sum / count;
        
        // Calculate Speed from the *De-Rotated* shift
        double speed_contribution = 1.0 * translation_shift * ang_res * depth;
        
        signed_speeds.push_back(speed_contribution);
        debug_good_rois.push_back(roi);
    }

    // -------------------------------------------------------------------------
    // 3. FUSION
    // -------------------------------------------------------------------------
    float final_speed = 0.0f;
    bool success = !signed_speeds.empty();

    if (success) {
        std::sort(signed_speeds.begin(), signed_speeds.end());
        double median_speed = signed_speeds[signed_speeds.size() / 2];
        double speed_sum = 0.0;
        int inlier_count = 0;
        for (float s : signed_speeds) {
            if (std::abs(s - median_speed) < 0.5) { 
                speed_sum += s;
                inlier_count++;
            }
        }
        final_speed = (inlier_count > 0) ? (speed_sum / inlier_count) : median_speed;
    }

    // -------------------------------------------------------------------------
    // 4. VISUALIZATION (Manual Gradient + Auto Phase)
    // -------------------------------------------------------------------------
    if (this->debug_phase_img_pub_->get_subscription_count() > 0) {
        
        cv::Mat raw_img;
        if (curr_img.type() == CV_8UC1) cv::cvtColor(curr_img, raw_img, cv::COLOR_GRAY2BGR);
        else curr_img.copyTo(raw_img);
        for (const auto& roi : debug_good_rois) cv::rectangle(raw_img, roi, cv::Scalar(0, 255, 0), 2);

        // Layer 2: Texture (Fixed Scale)
        cv::Mat energy_gray(1, w, CV_8UC1);
        const double FIXED_ENERGY_SCALE = 50000.0; 
        for (int i = 0; i < w; ++i) {
            double norm_val = std::min(vis_energy[i] / FIXED_ENERGY_SCALE, 1.0);
            energy_gray.at<uchar>(0, i) = static_cast<uchar>(norm_val * 255);
        }
        cv::Mat energy_color;
        cv::applyColorMap(energy_gray, energy_color, cv::COLORMAP_TURBO);
        cv::resize(energy_color, energy_color, raw_img.size(), 0, 0, cv::INTER_LINEAR);

        // Layer 3: Phase (Auto Norm)
        cv::Mat conf_strip(1, w, CV_64F, vis_confidence.data());
        cv::Mat conf_norm, conf_color;
        cv::normalize(conf_strip, conf_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(conf_norm, conf_color, cv::COLORMAP_VIRIDIS);
        cv::resize(conf_color, conf_color, raw_img.size(), 0, 0, cv::INTER_LINEAR);

        // Stack
        cv::Mat canvas;
        std::vector<cv::Mat> layers = {raw_img, energy_color, conf_color};
        cv::vconcat(layers, canvas);
        cv::resize(canvas, canvas, cv::Size(), 1.0, 3.0, cv::INTER_NEAREST); 

        // Text (Add Yaw info for debug)
        std::string status_txt = success ? "SPEED: " + std::to_string(final_speed).substr(0,5) + " m/s" : "NO LOCK";
        cv::Scalar status_color = success ? ((final_speed>0)?cv::Scalar(0,255,0):cv::Scalar(0,0,255)) : cv::Scalar(0,0,255);
        int h_scaled = h * 3.0; 
        
        auto drawLabel = [&](const std::string& text, cv::Point pt) {
            cv::putText(canvas, text, pt + cv::Point(2,2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
            cv::putText(canvas, text, pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        };
        drawLabel("1. RAW INTENSITY", cv::Point(10, 30));
        drawLabel("2. TEXTURE ENERGY (Fixed Scale)", cv::Point(10, h_scaled + 30));
        drawLabel("3. PHASE CONFIDENCE (Normalized)", cv::Point(10, h_scaled*2 + 30));

        // Show Rotation Correction Amount
        std::string rot_txt = "Rot Comp: " + std::to_string(rotation_shift_px).substr(0,4) + " px";
        drawLabel(rot_txt, cv::Point(w - 250, 30));

        int bottom_y = canvas.rows - 20;
        cv::rectangle(canvas, cv::Point(0, bottom_y - 60), cv::Point(600, bottom_y + 20), cv::Scalar(0,0,0), -1);
        cv::putText(canvas, status_txt, cv::Point(20, bottom_y), cv::FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4);

        std_msgs::msg::Header header;
        header.stamp = this->scan_header_stamp;
        header.frame_id = this->lidar_frame;
        this->debug_phase_img_pub_->publish(*cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg());
    }

    if (!success) return Eigen::Vector3f::Zero();

    return Eigen::Vector3f(final_speed, 0.0, 0.0);
}

// cv::Mat rls::OdomNode::generateIntensityImage(const pcl::PointCloud<PointType>::ConstPtr& cloud)
// {
//     // 1. Setup Images
//     cv::Mat intensity_img = cv::Mat::zeros(this->scan_H_, this->scan_W_, CV_32F);
//     cv::Mat valid_mask    = cv::Mat::zeros(this->scan_H_, this->scan_W_, CV_8UC1);

//     // 2. Fill Data
//     if (cloud->height > 1 && cloud->width > 1) {
//         for (int r = 0; r < cloud->height; r++) {
//             for (int c = 0; c < cloud->width; c++) {
//                 const auto& p = cloud->at(c, r);
                
//                 // Skip invalid points
//                 if (!std::isfinite(p.x) || (p.x==0 && p.y==0 && p.z==0)) continue;
                
//                 float val = p.intensity;

//                 // --- FIX: ROBUST SCALING STRATEGY ---
                
//                 // Step A: Log correction (Bring 10 and 2000 closer together)
//                 float log_val = std::log(val + 1.0f);

//                 // Step B: Manual Clamping for Ouster
//                 // Typical Ouster log values:
//                 // Road (10-50)  -> Log: 2.3 - 3.9
//                 // Walls (200)   -> Log: ~5.3
//                 // Retro (2000+) -> Log: ~7.6+
//                 // We set a "ceiling" at 8.0. Anything brighter is just "pure white".
//                 if (log_val > 8.0f) log_val = 8.0f;
                
//                 intensity_img.at<float>(r, c) = log_val;
//                 valid_mask.at<uint8_t>(r, c) = 255;
//             }
//         }
//     } 
//     // (Unordered fallback would go here)

//     // 3. Normalize to 0-255 using FIXED bounds
//     // Instead of asking "what is the max?", we TELL it "8.0 is the max".
//     // This guarantees the road (Log ~3.0) will be mapped to ~95 (Visible Gray), not 0.
//     cv::Mat img_8u;
//     intensity_img.convertTo(img_8u, CV_8UC1, 255.0 / 8.0); 

//     // 4. MASKED CLAHE (Texture Boosting)
//     // Now that the road is Gray (not Black), CLAHE can actually enhance its texture.
//     auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
//     clahe->apply(img_8u, img_8u);

//     // 5. Re-Apply Mask (Keep background black)
//     cv::Mat final_img;
//     img_8u.copyTo(final_img, valid_mask);

//     return final_img;
// }

cv::Mat rls::OdomNode::generateIntensityImage(const pcl::PointCloud<PointType>::ConstPtr& cloud)
{
    if (cloud->points.empty()) return cv::Mat();

    int H = cloud->height;
    int W = cloud->width;

    // -------------------------------------------------------------------------
    // CASE 1: ORGANIZED CLOUD (Ouster / Hesai)
    // -------------------------------------------------------------------------
    if (H > 1) {
        // [Existing Logic] - Check for resolution change
        if (this->scan_H_ != H || this->scan_W_ != W) {
            this->scan_H_ = H;
            this->scan_W_ = W;
        }

        cv::Mat intensity_img = cv::Mat::zeros(H, W, CV_32F);
        cv::Mat valid_mask = cv::Mat::zeros(H, W, CV_8UC1);

        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                const auto& p = cloud->at(c, r);
                if (!std::isfinite(p.x) || (p.x==0 && p.y==0 && p.z==0)) continue;
                
                float val = std::log(p.intensity + 1.0f);
                if (val > 8.0f) val = 8.0f;
                
                intensity_img.at<float>(r, c) = val;
                valid_mask.at<uint8_t>(r, c) = 255;
            }
        }
        
        cv::Mat img_8u;
        intensity_img.convertTo(img_8u, CV_8UC1, 255.0 / 8.0);
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img_8u, img_8u);
        
        cv::Mat final_img;
        img_8u.copyTo(final_img, valid_mask);
        return final_img;
    }

    // -------------------------------------------------------------------------
    // CASE 2: UNORGANIZED CLOUD (Velodyne VLP-32C)
    // -------------------------------------------------------------------------
    // We must manually project points into a 2D grid based on Azimuth/Elevation
    else {
        // Use params from YAML (e.g., 32 x 1800)
        int proj_H = this->scan_H_; 
        int proj_W = this->scan_W_;
        
        cv::Mat intensity_img = cv::Mat::zeros(proj_H, proj_W, CV_32F);
        cv::Mat valid_mask = cv::Mat::zeros(proj_H, proj_W, CV_8UC1);

        // Pre-compute vertical angle steps
        float fov_up_rad = (this->scan_fov_up_ * M_PI) / 180.0;
        float fov_down_rad = (this->scan_fov_down_ * M_PI) / 180.0;
        float fov_res = (std::abs(fov_up_rad) + std::abs(fov_down_rad)) / (proj_H - 1);

        for (const auto& p : cloud->points) {
            if (!std::isfinite(p.x) || (p.x==0 && p.y==0 && p.z==0)) continue;

            float range = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            if (range < 0.5) continue;

            // Calculate Angles
            float yaw = -std::atan2(p.y, p.x);
            float pitch = std::asin(p.z / range);

            // Project to Image Coordinates
            // 1. Column (Azimuth): Map -PI..PI to 0..W
            float proj_x = 0.5 * (yaw / M_PI + 1.0) * proj_W;

            // 2. Row (Elevation): Map FOV_DOWN..FOV_UP to H..0
            // Note: VLP-32 has non-linear spacing, but linear approx is okay for visual odometry
            float proj_y = 1.0 - (pitch + std::abs(fov_down_rad)) / (std::abs(fov_up_rad) + std::abs(fov_down_rad));
            proj_y *= proj_H;

            int u = std::clamp(static_cast<int>(proj_x), 0, proj_W - 1);
            int v = std::clamp(static_cast<int>(proj_y), 0, proj_H - 1);

            // Fill Pixel
            float val = std::log(p.intensity + 1.0f);
            if (val > 8.0f) val = 8.0f;

            intensity_img.at<float>(v, u) = val;
            valid_mask.at<uint8_t>(v, u) = 255;
        }

        cv::Mat img_8u;
        intensity_img.convertTo(img_8u, CV_8UC1, 255.0 / 8.0);
        
        // Dilate to fill gaps (Velodyne is sparse compared to Ouster)
        cv::dilate(img_8u, img_8u, cv::Mat(), cv::Point(-1, -1), 1);
        cv::dilate(valid_mask, valid_mask, cv::Mat(), cv::Point(-1, -1), 1);

        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img_8u, img_8u);

        cv::Mat final_img;
        img_8u.copyTo(final_img, valid_mask);
        return final_img;
    }
}

std::vector<cv::Point2f> rls::OdomNode::detectFeatures(const cv::Mat& img) {
    
    // 1. Create a Mask (White = Valid, Black = Ignore)
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // 2. Define Regions of Interest (ROI)
    // We want the LEFT WALL and the RIGHT WALL.
    // We want to ignore the CENTER (End of tunnel) and BOTTOM (Road blur).
    
    int w = img.cols;
    int h = img.rows;

    // Left Strip (0% to 40% width, Top 80% height)
    cv::Rect left_wall(0, 0, w * 0.40, h * 0.85);
    
    // Right Strip (60% to 100% width, Top 80% height)
    cv::Rect right_wall(w * 0.60, 0, w * 0.40, h * 0.85);

    // Draw white rectangles on the mask
    cv::rectangle(mask, left_wall, cv::Scalar(255), -1);
    cv::rectangle(mask, right_wall, cv::Scalar(255), -1);

    // 3. Apply Threshold to ignore dark pixels (existing logic)
    cv::Mat dark_mask;
    cv::threshold(img, dark_mask, 15, 255, cv::THRESH_BINARY);
    
    // Combine: Must be in ROI AND Bright enough
    cv::bitwise_and(mask, dark_mask, mask);

    // 4. Erode to be safe from edges
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(mask, mask, element);

    // 5. Detect
    std::vector<cv::KeyPoint> keypoints;
    this->detector_->detect(img, keypoints, mask);

    // 6. Bucketing (Keep your existing bucketing logic here...)
    // ... copy your existing grid/bucket loop here ...
    // (This ensures features are spread out evenly on the walls)
    
    // [Start of your existing bucketing code copy]
    int grid_cols = 16;
    int grid_rows = 4; 
    int cell_w = img.cols / grid_cols;
    int cell_h = img.rows / grid_rows;
    std::map<int, std::vector<cv::KeyPoint>> grid_features;
    for (const auto& kp : keypoints) {
        int c = static_cast<int>(kp.pt.x / cell_w);
        int r = static_cast<int>(kp.pt.y / cell_h);
        if (c >= grid_cols) c = grid_cols - 1;
        if (r >= grid_rows) r = grid_rows - 1;
        int grid_index = r * grid_cols + c;
        grid_features[grid_index].push_back(kp);
    }
    std::vector<cv::Point2f> final_points;
    int max_features_per_cell = 5; 
    for (auto& entry : grid_features) {
        auto& bucket = entry.second;
        std::sort(bucket.begin(), bucket.end(), 
                 [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                     return a.response > b.response; 
                 });
        for (size_t k = 0; k < bucket.size() && k < max_features_per_cell; k++) {
            final_points.push_back(bucket[k].pt);
        }
    }
    // [End of bucketing code]

    return final_points;
}

// std::vector<cv::Point2f> rls::OdomNode::detectFeatures(const cv::Mat& img) {
    
//     // 1. Create Safety Mask
//     cv::Mat mask;
//     // Identify valid pixels (anything brighter than deep black)
//     cv::threshold(img, mask, 10, 255, cv::THRESH_BINARY); 

//     // [NEW] Erode the mask to shrink the "safe zone"
//     // This forces features to be at least 5 pixels away from the black void.
//     // Kernel size 5 means "shave 2 pixels off every edge".
//     cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//     cv::erode(mask, mask, element);

//     // 2. Detect Globally (FAST + Eroded Mask)
//     std::vector<cv::KeyPoint> keypoints;
//     this->detector_->detect(img, keypoints, mask);

//     // 3. Bucketing (Same as before)
//     int grid_cols = 16;
//     int grid_rows = 4; 
//     int cell_w = img.cols / grid_cols;
//     int cell_h = img.rows / grid_rows;

//     std::map<int, std::vector<cv::KeyPoint>> grid_features;

//     for (const auto& kp : keypoints) {
//         int c = static_cast<int>(kp.pt.x / cell_w);
//         int r = static_cast<int>(kp.pt.y / cell_h);
        
//         if (c >= grid_cols) c = grid_cols - 1;
//         if (r >= grid_rows) r = grid_rows - 1;

//         int grid_index = r * grid_cols + c;
//         grid_features[grid_index].push_back(kp);
//     }

//     // 4. Select Best Features
//     std::vector<cv::Point2f> final_points;
//     int max_features_per_cell = 5; 

//     for (auto& entry : grid_features) {
//         auto& bucket = entry.second;
        
//         std::sort(bucket.begin(), bucket.end(), 
//                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
//                      return a.response > b.response; 
//                  });

//         for (size_t k = 0; k < bucket.size() && k < max_features_per_cell; k++) {
//             final_points.push_back(bucket[k].pt);
//         }
//     }

//     return final_points;
// }

Eigen::Matrix4f rls::OdomNode::computeVisualOdometry(
    const cv::Mat& curr_img,
    const std::vector<cv::Point2f>& curr_features,
    const pcl::PointCloud<PointType>::ConstPtr& /*unused_arg*/) 
{
    // Check if we have history AND the organized clouds
    if (this->prev_intensity_img_.empty() || this->prev_features_.empty() || !this->prev_full_organized_scan_) {
        return Eigen::Matrix4f::Identity();
    }

    // 1. Optical Flow
    std::vector<cv::Point2f> tracked_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(this->prev_intensity_img_, curr_img, 
                             this->prev_features_, tracked_pts, 
                             status, err, cv::Size(21, 21), 3);

    // 2. Convert to 3D Points using the ORGANIZED clouds
    std::vector<cv::Point3f> p3d_prev;
    std::vector<cv::Point3f> p3d_curr;

    int width = this->scan_W_; // Should be 1024

    // NEW: Define a "Reliable Range"
    // Points closer than 1.0m are often noisy (sensor near-field).
    // Points further than 20.0m have bad depth resolution and low parallax.
    // const float MIN_DEPTH = 1.0; 
    // const float MAX_DEPTH = 15.0;
    
    for (size_t i = 0; i < status.size(); i++) {
        if (!status[i]) continue;

        // Calculate 1D indices from 2D pixels
        int idx_prev = static_cast<int>(this->prev_features_[i].y) * width + static_cast<int>(this->prev_features_[i].x);
        int idx_curr = static_cast<int>(tracked_pts[i].y) * width + static_cast<int>(tracked_pts[i].x);

        // Safety Bounds Check
        if (idx_prev < 0 || idx_prev >= this->prev_full_organized_scan_->size() ||
            idx_curr < 0 || idx_curr >= this->full_organized_scan_->size()) {
            continue;
        }

        // [CRITICAL FIX] Lookup in the ORGANIZED clouds
        PointType pt_prev = this->prev_full_organized_scan_->points[idx_prev];
        PointType pt_curr = this->full_organized_scan_->points[idx_curr];

        // Valid Depth Check (Skip NaNs and points too close/far)
        if (std::isfinite(pt_prev.x) && std::abs(pt_prev.x) > 0.5 && 
            std::isfinite(pt_curr.x) && std::abs(pt_curr.x) > 0.5) {
            
            p3d_prev.push_back(cv::Point3f(pt_prev.x, pt_prev.y, pt_prev.z));
            p3d_curr.push_back(cv::Point3f(pt_curr.x, pt_curr.y, pt_curr.z));
        }

        // // [UPDATED] Valid Depth Check with Range Gating
        // // We reject points that are too far (infinite edge) or too close.
        // float d_prev = std::abs(pt_prev.x); // Assuming X is forward/depth in Lidar frame
        // float d_curr = std::abs(pt_curr.x);

        // if (std::isfinite(pt_prev.x) && d_prev > MIN_DEPTH && d_prev < MAX_DEPTH && 
        //     std::isfinite(pt_curr.x) && d_curr > MIN_DEPTH && d_curr < MAX_DEPTH) {
            
        //     p3d_prev.push_back(cv::Point3f(pt_prev.x, pt_prev.y, pt_prev.z));
        //     p3d_curr.push_back(cv::Point3f(pt_curr.x, pt_curr.y, pt_curr.z));
        // }
    }

    // 3. RANSAC Estimation (Required for Tunnels)
if (p3d_prev.size() < 10) {
        this->visual_inliers_ = 0; // [NEW] Reset count
        return Eigen::Matrix4f::Identity();
    }

    cv::Mat affine_out;
    std::vector<uchar> inliers;
    
    // Threshold 0.1m (Tight precision)
    int success = cv::estimateAffine3D(p3d_prev, p3d_curr, affine_out, inliers, 0.1, 0.99);

    // [NEW] Count the inliers
    this->visual_inliers_ = cv::countNonZero(inliers);

    if (!success) return Eigen::Matrix4f::Identity();

    // 4. Convert result
    // Eigen::Matrix4f T_visual = Eigen::Matrix4f::Identity();
    // for (int r = 0; r < 3; r++) {
    //     for (int c = 0; c < 4; c++) {
    //         T_visual(r, c) = static_cast<float>(affine_out.at<double>(r, c));
    //     }
    // }

    Eigen::Matrix4f T_visual = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            T_visual(r, c) = static_cast<float>(affine_out.at<double>(r, c));
        }
    }

    // [FIX] FORCE PURE TRANSLATION
    // Visual rotation is noisy. We trust the IMU for rotation. 
    // We only want the "Forward Shift" from vision.
    T_visual.block<3,3>(0,0) = Eigen::Matrix3f::Identity();

    // 5. Sanity Check
    float mag = T_visual.block<3,1>(0,3).norm();
    if (mag > 2.0 || std::isnan(mag)) return Eigen::Matrix4f::Identity();

    return T_visual;
}

// Eigen::Matrix4f rls::OdomNode::computeVisualOdometry(
//     const cv::Mat& curr_img,
//     const std::vector<cv::Point2f>& curr_features,
//     const pcl::PointCloud<PointType>::ConstPtr& curr_cloud)
// {
//     if (this->prev_intensity_img_.empty() || this->prev_features_.empty()) {
//         return Eigen::Matrix4f::Identity();
//     }

//     std::vector<cv::Point2f> tracked_pts;
//     std::vector<uchar> status;
//     std::vector<float> err;

//     cv::calcOpticalFlowPyrLK(this->prev_intensity_img_, curr_img, 
//                              this->prev_features_, tracked_pts, 
//                              status, err, cv::Size(21, 21), 3);

//     pcl::PointCloud<PointType>::Ptr pts_prev(new pcl::PointCloud<PointType>());
//     pcl::PointCloud<PointType>::Ptr pts_curr(new pcl::PointCloud<PointType>());

//     int width = this->scan_W_;
//     int height = this->scan_H_;

//     for (size_t i = 0; i < status.size(); i++) {
//         if (!status[i]) continue;

//         cv::Point2f p_prev_2d = this->prev_features_[i];
//         cv::Point2f p_curr_2d = tracked_pts[i];

//         // [FIXED] Use static_cast to remove warnings
//         int idx_prev = static_cast<int>(p_prev_2d.y) * width + static_cast<int>(p_prev_2d.x);
//         int idx_curr = static_cast<int>(p_curr_2d.y) * width + static_cast<int>(p_curr_2d.x);

//         if (idx_prev >= 0 && idx_prev < static_cast<int>(this->prev_original_scan_->size()) &&
//             idx_curr >= 0 && idx_curr < static_cast<int>(curr_cloud->size())) {
            
//             PointType pt3d_prev = this->prev_original_scan_->points[idx_prev];
//             PointType pt3d_curr = curr_cloud->points[idx_curr];

//             if (std::isfinite(pt3d_prev.x) && pt3d_prev.x != 0 &&
//                 std::isfinite(pt3d_curr.x) && pt3d_curr.x != 0) {
                
//                 pts_prev->push_back(pt3d_prev);
//                 pts_curr->push_back(pt3d_curr);
//             }
//         }
//     }

//   if (pts_prev->size() < 10) {
//         return Eigen::Matrix4f::Identity();
//     }

//     pcl::registration::TransformationEstimationSVD<PointType, PointType> svd;
//     Eigen::Matrix4f T_visual;
//     svd.estimateRigidTransformation(*pts_prev, *pts_curr, T_visual);

//     // [NEW] Sanity Check: If the transform is crazy large, reject it.
//     // e.g., if it says we moved > 1.0 meter in 0.1 seconds (10 m/s), it's probably noise.
//     float translation_mag = T_visual.block<3,1>(0,3).norm();
//     if (translation_mag > 1.0) { 
//         return Eigen::Matrix4f::Identity(); // Reject outlier
//     }

//     return T_visual;
// }

cv::Mat rls::OdomNode::filterRangeImage(const cv::Mat& range_img)
{
    // Expect CV_32F range image
    CV_Assert(range_img.type() == CV_32F);

    const int rows = range_img.rows;
    const int cols = range_img.cols;

    // 1) Build mask of valid pixels (range > 0)
    cv::Mat valid_mask = cv::Mat::zeros(rows, cols, CV_8U);
    for (int r = 0; r < rows; ++r)
    {
        const float* row_ptr = range_img.ptr<float>(r);
        uchar* mask_ptr = valid_mask.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c)
        {
            if (row_ptr[c] > 0.05f)  // ignore zeros / very tiny
                mask_ptr[c] = 255;
        }
    }

    // 2) Compute min/max only over valid pixels
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(range_img, &minVal, &maxVal, nullptr, nullptr, valid_mask);

    // Safety clamp
    if (maxVal <= minVal + 1e-6)
        maxVal = minVal + 1.0;

    // 3) Normalize to [0,255], optionally invert so near objects = bright
    cv::Mat norm_8u(rows, cols, CV_8U, cv::Scalar(0));

    for (int r = 0; r < rows; ++r)
    {
        const float* row_ptr = range_img.ptr<float>(r);
        uchar* out_ptr = norm_8u.ptr<uchar>(r);
        uchar* mask_ptr = valid_mask.ptr<uchar>(r);

        for (int c = 0; c < cols; ++c)
        {
            if (!mask_ptr[c]) {
                out_ptr[c] = 0;  // invalid → black
                continue;
            }

            float d = row_ptr[c];
            // Normalize: far → dark, near → bright (invert depth)
            float alpha = static_cast<float>((d - minVal) / (maxVal - minVal));
            alpha = std::clamp(alpha, 0.0f, 1.0f);
            float inv = 1.0f - alpha;
            out_ptr[c] = static_cast<uchar>(inv * 255.0f);
        }
    }

    // 4) Edge-preserving smoothing (bilateral filter)
    cv::Mat smooth;
    // Parameters can be tuned; 9/75/5 is often good for depth-like images
    cv::bilateralFilter(norm_8u, smooth, /*diameter*/ 9, /*sigmaColor*/ 75.0, /*sigmaSpace*/ 5.0);

    // 5) CLAHE for local contrast enhancement (brings out texture)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);                   // adjust if too strong/weak
    clahe->setTilesGridSize(cv::Size(8, 8));    // local window size

    cv::Mat clahe_out;
    clahe->apply(smooth, clahe_out);

    // 6) Colorize using a colormap (JET or TURBO)
    cv::Mat colored;
    cv::applyColorMap(clahe_out, colored, cv::COLORMAP_TURBO);
    // Alternative: cv::COLORMAP_JET, cv::COLORMAP_PLASMA, etc.

    // Optional: mask out invalid pixels to black in colored output
    for (int r = 0; r < rows; ++r)
    {
        uchar* mask_ptr = valid_mask.ptr<uchar>(r);
        cv::Vec3b* col_ptr = colored.ptr<cv::Vec3b>(r);
        for (int c = 0; c < cols; ++c)
        {
            if (!mask_ptr[c]) {
                col_ptr[c] = cv::Vec3b(0, 0, 0);  // black where no data
            }
        }
    }

    return colored;  // CV_8UC3, good for RViz / display
}



void rls::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();
    }

  }

}

void rls::OdomNode::getNextPose() {

  // Save the pose BEFORE GICP (T_k-1)
  Eigen::Matrix4f T_start = this->T; 

  // --------------------------------------------------------------------------
  // [FIX 1] GICP REGISTRATION SAFETY
  // --------------------------------------------------------------------------
  // Check if future is valid
  if (this->submap_future.valid()) {
      // If map is empty (first frame), WAIT for it.
      if (this->submap_cloud->empty()) {
          this->submap_future.wait();
          this->new_submap_is_ready = true; 
      } else {
          // Otherwise just check status
          this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
      }
  }

  // FORCE REGISTRATION: We removed '&& submap_hasChanged'. 
  // If we have a map, we MUST register it, or GICP crashes.
  if (this->new_submap_is_ready && !this->submap_cloud->empty()) {
      this->gicp.registerInputTarget(this->submap_cloud);
      this->gicp.target_kdtree_ = this->submap_kdtree;
      this->gicp.setTargetCovariances(this->submap_normals);
      this->submap_hasChanged = false;
  }
  // --------------------------------------------------------------------------

  // [FIX 2] RUN GICP (Only if target exists)
  if (!this->submap_cloud->empty()) {
      pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
      this->gicp.align(*aligned); // This will now work
      this->T_corr = this->gicp.getFinalTransformation();

      // =================================================================
      // 🚨 METRICS EXTRACTION FOR PAPER (Odometry FFT) 🚨
      // =================================================================
      if (this->gicp.hasConverged()) {
          
          // Grab the raw frame-to-frame translation step
          float step_x = this->T_corr(0,3);
          float step_y = this->T_corr(1,3);
          float step_z = this->T_corr(2,3);

          // Change to "trajectory_jitter_standard.csv" when use_vibration_mask_ = false
          std::string log_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/trajectory_jitter_scd.csv";
          // std::string log_path = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/trajectory_jitter_standard.csv";
          std::ofstream log_file(log_path, std::ios_base::app);
          if (log_file.is_open()) {
              log_file << std::fixed << std::setprecision(6) << this->scan_stamp << "," << step_x << "," << step_y << "," << step_z << "\n";
              log_file.close();
          }
      }
      // =================================================================
  } else {
      this->T_corr = Eigen::Matrix4f::Identity(); // Fallback if no map yet
  }
  
  // Calculate where GICP thinks we are
  Eigen::Matrix4f T_gicp_final = this->T_corr * this->T_prior;

  // --------------------------------------------------------------------------
  // [FIX 3] COMPONENT REPLACEMENT FUSION (X-Only)
  // --------------------------------------------------------------------------
  // if (this->visual_valid_) {
      
  //     // A. Extract "Body Motion" from GICP
  //     // "How much did GICP move us relative to the previous frame?"
  //     // T_final = T_start * T_delta  =>  T_delta = T_start^-1 * T_final
  //     Eigen::Matrix4f T_delta_body = T_start.inverse() * T_gicp_final;

  //     // B. The Surgical Swap
  //     // T_delta_body(0,3) is GICP's Forward motion (Degenerate/Sliding)
  //     // T_delta_body(1,3) is GICP's Lateral motion (Accurate)
  //     // T_delta_body(2,3) is GICP's Vertical motion (Accurate)
  //     // Rotation block is GICP's Rotation (Accurate)
      
  //     // We overwrite ONLY the X (Forward) component with Vision
  //     T_delta_body(0,3) = this->visual_body_vel_smoothed_.x();

  //     // C. Re-Integrate to Global Frame
  //     this->T = T_start * T_delta_body;

  //     // D. Update Correction Matrix (Maintain internal consistency)
  //     this->T_corr = this->T * this->T_prior.inverse();
      
  //     this->visual_valid_ = false;
  // } 
  // else {
  //     // Vision failed? Trust GICP fully
  //     this->T = T_gicp_final;
  // }
  // --------------------------------------------------------------------------

  if (this->visual_valid_) {
    // Calculate Body Delta
    Eigen::Matrix4f T_delta_body = T_start.inverse() * T_gicp_final;
    
    // Swap X
    T_delta_body(0,3) = this->visual_body_vel_smoothed_.x();
    
    // Re-Integrate
    this->T = T_start * T_delta_body;
    
    // Update Correction
    this->T_corr = this->T * this->T_prior.inverse();
    this->visual_valid_ = false;
  } 
  else {
      // Vision failed? Trust GICP fully
      this->T = T_gicp_final;
  }

  this->propagateGICP();
  this->updateState();
}

// void rls::OdomNode::getNextPose() {

//   // Check if the new submap is ready to be used
//   this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

//   if (this->new_submap_is_ready && this->submap_hasChanged) {

//     // Set the current global submap as the target cloud
//     this->gicp.registerInputTarget(this->submap_cloud);

//     // Set submap kdtree
//     this->gicp.target_kdtree_ = this->submap_kdtree;

//     // Set target cloud's normals as submap normals
//     this->gicp.setTargetCovariances(this->submap_normals);

//     this->submap_hasChanged = false;
//   }

//   Eigen::Vector3f pos_prior = this->T_prior.block<3,1>(0,3);

//   // Run GICP
//   pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
//   this->gicp.align(*aligned);
//   this->T_corr = this->gicp.getFinalTransformation(); 
//   this->T = this->T_corr * this->T_prior;

//   // [FIX] TUNNEL LOGIC: Trust Vision if Valid
//   if (this->visual_valid_) {
//       // Trust Vision for Position (Stop GICP sliding)
//       // Keep GICP Rotation
//       this->T.block<3,1>(0,3) = pos_prior; 
      
//       this->visual_valid_ = false;
//   }

//   this->propagateGICP();
//   this->updateState();
// }

// void rls::OdomNode::getNextPose() {

//   // Check if the new submap is ready to be used
//   this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

//   if (this->new_submap_is_ready && this->submap_hasChanged) {

//     // Set the current global submap as the target cloud
//     this->gicp.registerInputTarget(this->submap_cloud);

//     // Set submap kdtree
//     this->gicp.target_kdtree_ = this->submap_kdtree;

//     // Set target cloud's normals as submap normals
//     this->gicp.setTargetCovariances(this->submap_normals);

//     this->submap_hasChanged = false;
//   }

//   Eigen::Vector3f pos_prior = this->T_prior.block<3,1>(0,3);

//   // Align with current submap with global IMU transformation as initial guess
//   pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
//   this->gicp.align(*aligned);

//   // Get final transformation in global frame
//   this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
//   this->T = this->T_corr * this->T_prior;

//   Eigen::Vector3f pos_final = this->T.block<3,1>(0,3);

//   // // 4. CALCULATE THE CONFLICT
//   // // This vector represents "How much GICP moved us relative to the Prior"
//   Eigen::Vector3f gicp_fight = pos_final - pos_prior;

//   // 5. PRINT DEBUGGING
//   // If 'gicp_fight.x' is NEGATIVE while you move forward, GICP is the problem.
//   RCLCPP_INFO(this->get_logger(), 
//       "\n[SOLVER FIGHT DEBUG]\n"
//       "  Visual Guess (Prior): %.4f\n"
//       "  GICP Result (Final):  %.4f\n"
//       "  GICP CORRECTION:      %.4f  <-- (If this is negative, GICP is pulling you back)",
//       pos_prior.x(), 
//       pos_final.x(),
//       gicp_fight.x() // Assuming X is forward in your global frame
//   );

//   // // --------------------------------------------------------------------------
//   // [NEW] SMART FUSION / DEGENERACY HANDLING
//   // --------------------------------------------------------------------------
//   if (this->visual_valid_) {
      
//       // Eigen::Vector3f pos_final = this->T.block<3,1>(0,3);
      
//       // // Calculate how much GICP fought against Vision
//       // // We project the error onto the "Forward" direction (Body X)
//       // // (Simplified: Just check total distance for now)
//       // float conflict_dist = (pos_final - pos_prior).norm();
      
//       // // THRESHOLD: 0.15 meters (15 cm)
//       // // If GICP moves us < 15cm, it's likely just fixing sensor noise.
//       // // If GICP moves us > 15cm, it is likely SLIDING down the tunnel.
      
//       // if (conflict_dist > 0.15) {
          
//       //     RCLCPP_WARN(this->get_logger(), "Tunnel Slip Detected! (Conflict: %.2fm). Forcing Visual Position.", conflict_dist);
          
//       //     // FORCE STRATEGY:
//       //     // Keep GICP Rotation (It knows 'Down' and 'Turn' best)
//       //     // Keep Visual Translation (It knows 'Forward' best)
          
//       //     this->T.block<3,1>(0,3) = pos_prior; 
          
//       /////////////////////////////////////////////////////////////
//       // } else {
//       //     // NORMAL STRATEGY:
//       //     // GICP agrees roughly with Vision. Trust GICP's refinement.
//       //     // (No Action needed, let T stay as is)
//       // }

//       // // Reset flag
//       // this->visual_valid_ = false;

//       // We DO NOT check if the conflict is > 0.15. 
//       // We overwrite GICP's translation because GICP is drifting in the tunnel.
      
//       // 1. Keep GICP Rotation (Best for keeping robot flat)
//       // 2. Force Visual Position (Best for moving forward)
//       this->T.block<3,1>(0,3) = pos_prior; 
      
//       // Reset flag immediately
//       this->visual_valid_ = false;
//   }
//   // --------------------------------------------------------------------------
//   // std::cout << "|Current Transformation :  = " << this->T << " " << " |Prior Transformation" << this->T_prior << std::endl;

//   // Update next global pose
//   this->propagateGICP();

//   // Geometric observer update
//   this->updateState();

// }

bool rls::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
rls::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
rls::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;
    prev_imu_it = imu_it;
  }

  return imu_se3;

}

void rls::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void rls::OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void rls::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;
  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

// Testing
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
rls::OdomNode::integrateImuTest(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  // v_init -= a1*idt + 0.5*j*idt*idt;

  if (abs(a1[0]) >= 0.25 || abs(a2[1] >= 0.25)) {
    p_init -=  v_init*idt;
  } else {
     v_init -= a1*idt + 0.5*j*idt*idt;
     p_init -= v_init*idt + 0.5*a1*idt*idt;
  }

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

sensor_msgs::msg::Imu::SharedPtr rls::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  static double prev_stamp = imu_stamp_secs;
  double dt = imu_stamp_secs - prev_stamp;
  prev_stamp = imu_stamp_secs;
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void rls::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

// void rls::OdomNode::computeSpaciousness() {

//   // compute range of points
//   std::vector<float> ds;

//   for (int i = 0; i <= this->original_scan->points.size(); i++) {
//     float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
//                         pow(this->original_scan->points[i].y, 2));
//     ds.push_back(d);
//   }

//   // median
//   std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
//   float median_curr = ds[ds.size()/2];
//   static float median_prev = median_curr;
//   float median_lpf = 0.95*median_prev + 0.05*median_curr;
//   median_prev = median_lpf;

//   // push
//   this->metrics.spaciousness.push_back( median_lpf );

// }

void rls::OdomNode::computeSpaciousness() {

  if (this->align_paramas_) {
    std::vector<float> maha_ds;
    float mahalanobis_total = 0;
    float num_maha = 0;
    for (int i = 0; i <= this->original_scan->points.size(); i++){
        float d_x = this->original_scan->points[i].x - this->deskewed_scan->points[i].x; float d_y = this->original_scan->points[i].y - this->deskewed_scan->points[i].y;
        float d_z = this->original_scan->points[i].z - this->deskewed_scan->points[i].z;
        Eigen::Vector3d d_diff (this->original_scan->points[i].x - this->deskewed_scan->points[i].x,
                                this->original_scan->points[i].y - this->deskewed_scan->points[i].y,
                                this->original_scan->points[i].z - this->deskewed_scan->points[i].z);

        if (this->mahalanobis_idx_){
            Eigen::Matrix4d last_cov = this->submap_normals->back();
            Eigen::Matrix3d lastest_covariance = last_cov.block<3,3>(0,0);
            Eigen::Matrix3d covariance_inv = lastest_covariance.inverse();

            float mahalanobis_idx = sqrt(d_diff.transpose() * covariance_inv * d_diff);
            maha_ds.push_back(mahalanobis_idx_);
            mahalanobis_total += mahalanobis_idx;
            num_maha++;
        }
    }

    // Median
    if (this->mahalanobis_idx_){
        std::nth_element(maha_ds.begin(), maha_ds.begin() + maha_ds.size()/2, maha_ds.end());
        float median_curr_maha = mahalanobis_total/num_maha;
        median_curr_maha /= 50;
        if (median_curr_maha > 100) {
            median_curr_maha = 0.5;
        }
        // std::cout << "Mahalanobis index : = " << " " << median_curr_maha << std::endl;
        static float median_prev = median_curr_maha;
        float median_lpf = 0.95*median_prev + 0.05*median_curr_maha;
        this->metrics.spaciousness.push_back(median_lpf);
    }
  }
  else {
    // compute range of points
    std::vector<float> ds;

    for (int i = 0; i <= this->original_scan->points.size(); i++) {
      float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                          pow(this->original_scan->points[i].y, 2));
      ds.push_back(d);
    }

    // median
    std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
    float median_curr = ds[ds.size()/2];
    static float median_prev = median_curr;
    float median_lpf = 0.95*median_prev + 0.05*median_curr;
    median_prev = median_lpf;

    // push
    this->metrics.spaciousness.push_back( median_lpf );
  }
  

}


void rls::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );

}

void rls::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void rls::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void rls::OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  if (this->align_paramas_) {
    for (const auto& k : this->keyframes) {
      // Get last covariance matrix a
      Eigen::Matrix4d last_cov = this->submap_normals->back();
      Eigen::Matrix3d lastest_covariance = last_cov.block<3,3>(0,0);
      Eigen::Matrix3d covariance_inv = lastest_covariance.inverse();
      Eigen::Vector3d diff = this->state.p.cast<double>() - k.first.first.cast<double>();
      float mahalanobis_dist = sqrt(diff.transpose() * covariance_inv * diff);
      // count the number nearby current pose
      if (mahalanobis_dist <= this->keyframe_thresh_dist_ * 1.2f){
        ++num_nearby;
      }

      // store into variable
      if (mahalanobis_dist < closest_d) {
        closest_d = mahalanobis_dist;
        closest_idx = keyframes_idx;
      }

      keyframes_idx++;
      this->mahalanobis_idx_ = true;
    }
  } else {
      for (const auto& k : this->keyframes) {

        // calculate distance between current pose and pose in keyframes
        float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                              pow(this->state.p[1] - k.first.first[1], 2) +
                              pow(this->state.p[2] - k.first.first[2], 2) );

        // count the number nearby current pose
        if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
          ++num_nearby;
        }

        // store into variable
        if (delta_d < closest_d) {
          closest_d = delta_d;
          closest_idx = keyframes_idx;
        }

        keyframes_idx++;
        this->mahalanobis_idx_ = true;
      }
  }


  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.push_back(this->T_corr);
    lock.unlock();

  }

}

void rls::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void rls::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void rls::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Modification
    Eigen::Matrix4d mat_curr = this->submap_normals->back();
    Eigen::Matrix3d mat_cov  = mat_curr.block<3,3>(0,0);

    // Convert to float
    Eigen::Matrix3f mat_cov_fl = mat_cov.cast<float>();
    this->R = mat_cov_fl;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void rls::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publish_keyframe_thread = std::thread( &rls::OdomNode::publishKeyframe, this, this->keyframes[i], this->keyframe_timestamps[i] );
    this->publish_keyframe_thread.detach();
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void rls::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void rls::OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // // Saving odometry in strict TUM format
  // std::ofstream outFile("/home/ara/ros2_ws/src/robust_lidar_inertial/log/trajectory.txt", std::ios::app);
  // if (!outFile.is_open()) {
  //     RCLCPP_ERROR(this->get_logger(), "Failed to open trajectory file!");
  //     return;
  // }

  // // Get timestamp in seconds.nanoseconds format
  // int64_t total_ns = this->imu_stamp.nanoseconds();
  // int64_t sec = total_ns / 1000000000;
  // int64_t nsec = total_ns % 1000000000;

  // // Write in strict TUM format (8 space-separated values)
  // outFile << sec << "." << std::setw(9) << std::setfill('0') << nsec << " " 
  //         << std::setprecision(9) << this->state.p[0] << " " 
  //         << this->state.p[1] << " " 
  //         << this->state.p[2] << " "
  //         << this->state.q.x() << " "
  //         << this->state.q.y() << " "
  //         << this->state.q.z() << " "
  //         << this->state.q.w() << "\n";  // Use \n instead of endl to avoid unnecessary flush

  // outFile.close();

  // // Print to terminal
  // printf("\033[2J\033[1;1H");
  // std::time_t curr_time = this->scan_stamp;
  // std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  // std::cout << "| " << std::left << asc_time;
  // std::cout << std::right << std::setfill(' ') << std::setw(42)
  //   << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
  //   << "|" << std::endl;

  // if ( !this->cpu_type.empty() ) {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << this->cpu_type + " x " + std::to_string(this->numProcessors)
  //     << "|" << std::endl;
  // }

  // if (this->sensor == rls::SensorType::OUSTER) {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
  //                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
  //     << "|" << std::endl;
  // } else if (this->sensor == rls::SensorType::VELODYNE) {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
  //                                    + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
  //     << "|" << std::endl;
  // } else if (this->sensor == rls::SensorType::HESAI) {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
  //                                 + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
  //     << "|" << std::endl;
  // } else if (this->sensor == rls::SensorType::LIVOX) {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
  //                                 + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
  //     << "|" << std::endl;
  // } else {
  //   std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //     << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
  //                                         + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
  //     << "|" << std::endl;
  // }

  // std::cout << "|===================================================================|" << std::endl;

  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
  //                               + to_string_with_precision(this->state.p[1], 4) + " "
  //                               + to_string_with_precision(this->state.p[2], 4)
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
  //                               + to_string_with_precision(this->state.q.x(), 4) + " "
  //                               + to_string_with_precision(this->state.q.y(), 4) + " "
  //                               + to_string_with_precision(this->state.q.z(), 4)
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
  //                               + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
  //                               + to_string_with_precision(this->state.v.lin.b[2], 4)
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
  //                               + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
  //                               + to_string_with_precision(this->state.v.ang.b[2], 4)
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
  //                               + to_string_with_precision(this->state.b.accel[1], 8) + " "
  //                               + to_string_with_precision(this->state.b.accel[2], 8)
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
  //                               + to_string_with_precision(this->state.b.gyro[1], 8) + " "
  //                               + to_string_with_precision(this->state.b.gyro[2], 8)
  //   << "|" << std::endl;

  // std::cout << "|                                                                   |" << std::endl;

  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Distance to Origin :: "
  //     + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
  //                                      pow(this->state.p[1]-this->origin[1],2) +
  //                                      pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
  //   << "|" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
  //                              + "deskewed points: " + std::to_string(this->deskew_size)
  //   << "|" << std::endl;
  // std::cout << "|                                                                   |" << std::endl;

  // std::cout << std::right << std::setprecision(2) << std::fixed;
  // std::cout << "| Computation Time :: "
  //   << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
  //   << std::setw(6) << avg_comp_time*1000. << " / Max: "
  //   << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
  //   << "     |" << std::endl;
  // std::cout << "| Cores Utilized   :: "
  //   << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
  //   << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
  //   << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
  //                      * this->numProcessors
  //   << "     |" << std::endl;
  // std::cout << "| CPU Load         :: "
  //   << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
  //   << std::setw(6) << avg_cpu_usage << " / Max: "
  //   << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
  //   << "     |" << std::endl;
  // std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
  //   << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
  //   << "|" << std::endl;

  // std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
