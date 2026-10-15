cmake_minimum_required(VERSION 3.22)
project(flytrack_camera_timing LANGUAGES CXX)
find_package(gz-sensors 10.0.2 EXACT REQUIRED COMPONENTS rendering)
find_package(gz-msgs REQUIRED)
find_package(gz-transport REQUIRED)
find_package(gz-rendering REQUIRED)
find_package(gz-common REQUIRED COMPONENTS graphics profiler)
add_library(flytrack_camera_timing SHARED
  "${CAMERA_SOURCE_DIR}/CameraSensor.cc"
  "${CAMERA_SOURCE_DIR}/CameraSensorUtil.cc")
target_compile_features(flytrack_camera_timing PRIVATE cxx_std_17)
target_compile_definitions(flytrack_camera_timing PRIVATE CameraSensor_EXPORTS)
target_link_libraries(flytrack_camera_timing PRIVATE
  gz-sensors::rendering gz-msgs::gz-msgs gz-transport::gz-transport
  gz-rendering::gz-rendering gz-common::graphics gz-common::profiler)
set_target_properties(flytrack_camera_timing PROPERTIES
  OUTPUT_NAME gz-sensors-camera VERSION 10.0.2 SOVERSION 10)
install(TARGETS flytrack_camera_timing LIBRARY DESTINATION lib)
install(FILES "${CAMERA_SOURCE_DIR}/source_manifest.json" DESTINATION share/flytrack_camera_timing)
