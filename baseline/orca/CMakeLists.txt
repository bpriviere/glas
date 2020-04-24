cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project (Orca)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options system REQUIRED)

add_subdirectory(RVO2)

# find_package (Eigen3 REQUIRED)
# find_package(stdc++fs)

include_directories(
  RVO2/src
  ${Boost_INCLUDE_DIRS}
)

add_executable(orca
  orca.cpp
)
target_link_libraries(orca
  RVO
  ${Boost_LIBRARIES}
  yaml-cpp
)
