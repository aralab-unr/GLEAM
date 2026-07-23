#include "rls/map.h"
#include "rls/utils.h"

rls::MapNode::MapNode(): Node("rls_map_node") {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&rls::MapNode::callbackKeyframe, this, std::placeholders::_1), keyframe_sub_opt);

  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map", 100);

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<robust_lidar_inertial::srv::SavePCD>("save_pcd",
      std::bind(&rls::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, this->save_pcd_cb_group);

  this->rls_map = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

}

rls::MapNode::~MapNode() {}

void rls::MapNode::getParams() {

  this->declare_parameter<std::string>("odom/odom_frame", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);

  this->get_parameter("odom/odom_frame", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);
}

void rls::MapNode::start() {
}

void rls::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // convert scan to pcl format
  pcl::PointCloud<PointType>::Ptr keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // voxel filter
  this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
  this->voxelgrid.setInputCloud(keyframe_pcl);
  this->voxelgrid.filter(*keyframe_pcl);

  // save filtered keyframe to map for rviz
  *this->rls_map += *keyframe_pcl;

  // publish full map
  if (this->rls_map->points.size() == this->rls_map->width * this->rls_map->height) {
    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*this->rls_map, map_ros);
    map_ros.header.stamp = this->now();
    map_ros.header.frame_id = this->odom_frame;
    this->map_pub->publish(map_ros);
  } 
}

void rls::MapNode::savePCD(std::shared_ptr<robust_lidar_inertial::srv::SavePCD::Request> req,
                            std::shared_ptr<robust_lidar_inertial::srv::SavePCD::Response> res) {

  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->rls_map);
  float leaf_size = req->leaf_size;
  std::string p = req->save_path;

  std::cout << std::setprecision(2) << "Saving map to " << p + "/rls_map.pcd"
    << " with leaf size " << to_string_with_precision(leaf_size, 2) << "... "; std::cout.flush();

  // voxelize map
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.setInputCloud(m);
  vg.filter(*m);

  // save map
  int ret = pcl::io::savePCDFileBinary(p + "/rls_map.pcd", *m);
  res->success = ret == 0;

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}
