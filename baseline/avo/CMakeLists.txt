cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project (Avo)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options system REQUIRED)

add_subdirectory(AVO2)

# find_package (Eigen3 REQUIRED)
# find_package(stdc++fs)

include_directories(
  AVO2/src
  ${Boost_INCLUDE_DIRS}
)

add_executable(avo
  main.cpp
)
target_link_libraries(avo
  AVO
  ${Boost_LIBRARIES}
  yaml-cpp
)
