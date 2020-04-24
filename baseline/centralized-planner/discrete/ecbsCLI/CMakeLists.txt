cmake_minimum_required(VERSION 2.8)

project(ecbsCLI)

# set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options REQUIRED)
FIND_PACKAGE( Eigen3 )
FIND_PACKAGE(yaml-cpp REQUIRED)

add_definitions(
  -DNDEBUG
)

# Additional include folders
include_directories(
  ../libCommon/include
  ../libEcbs/include
  ${EIGEN3_INCLUDE_DIR}
  /home/whoenig/sw/dlib-19.2
)

add_executable(ecbsCLI
  src/main.cpp
)

target_link_libraries(ecbsCLI
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  libCommon
  libEcbs
)

target_compile_options(ecbsCLI PUBLIC -std=c++11 -Wall)
