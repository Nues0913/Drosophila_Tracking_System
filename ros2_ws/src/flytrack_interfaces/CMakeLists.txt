cmake_minimum_required(VERSION 3.20)
project(flytrack_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OdorReading.msg" "msg/NeuralActivity.msg"
  "msg/TrackingMetrics.msg" "msg/ScenarioEvent.msg"
  "msg/SensorTiming.msg" "msg/ControlTrace.msg"
  "msg/WindReading.msg"
  DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
