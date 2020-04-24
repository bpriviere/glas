cmake_minimum_required(VERSION 2.8.3)
project(libEcbs)

FIND_PACKAGE( Boost 1.58 REQUIRED )
FIND_PACKAGE( Eigen3 )

# Additional include folders
include_directories(
  include/libEcbs
  ../libCommon/include
  src/
  ${EIGEN3_INCLUDE_DIR}
)

add_definitions(
  -DNDEBUG
)

## Declare a cpp library
add_library(libEcbs
  src/ecbs.cpp
  src/compute_heuristic.cpp
  src/ecbs_node.cpp
  src/ecbs_search.cpp
  src/node.cpp
  src/single_agent_ecbs.cpp
)

# Specify libraries to link a library or executable target against
target_link_libraries(libEcbs
  libCommon
)

