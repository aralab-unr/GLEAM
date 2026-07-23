#include "rls/map.h"

int main(int argc, char** argv) {

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rls::MapNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();

  return 0;

}
