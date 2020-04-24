cmake_minimum_required(VERSION 2.8.3)
project(libCommon)

FIND_PACKAGE( Eigen3 )
FIND_PACKAGE(yaml-cpp REQUIRED)

# Additional include folders
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

## Declare a cpp library
add_library(libCommon
  src/map.cpp
  src/AgentsLoaderYaml.cpp
  src/searchgraph.cpp
)

# Specify libraries to link a library or executable target against
target_link_libraries(libCommon
  ${YAML_CPP_LIBRARIES}
)

