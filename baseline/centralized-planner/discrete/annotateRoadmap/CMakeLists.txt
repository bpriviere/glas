cmake_minimum_required(VERSION 2.8)

project(annotateRoadmap)

# set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options filesystem REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
#FIND_PACKAGE(Threads REQUIRED)
#FIND_PACKAGE(fcl REQUIRED)
#FIND_PACKAGE(assimp REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)
#FIND_PACKAGE(octomap REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#===============================================================================
# Find required dependency libccd
#
# If libccd is not found, manually set the cache variables CCD_INCLUDE_DIR and
# CCD_LIBRARY
#===============================================================================
find_package(ccd QUIET)

# If ccd-config.cmake is not found, use pkg-config and/or find_path() and
# find_library()
if(NOT ccd_FOUND)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_CCD ccd)
    pkg_check_modules(PC_LIBCCD libccd)
  endif()

  find_path(CCD_INCLUDE_DIR ccd/ccd.h
    HINTS "${PC_CCD_INCLUDE_DIRS}" "${PC_LIBCCD_INCLUDE_DIRS}")

  # Using find_library() even if pkg-config is available ensures that the full
  # path to the ccd library is available in CCD_LIBRARIES
  find_library(CCD_LIBRARY ccd
    HINTS "${PC_CCD_LIBRARY_DIRS}" "${PC_LIBCCD_LIBRARY_DIRS}")

  # libccd links to LibM on UNIX.
  if(CYGWIN OR NOT WIN32)
    find_library(M_LIBRARY m)
  endif()

  if(CCD_INCLUDE_DIR AND CCD_LIBRARY)
    set(CCD_INCLUDE_DIRS "${CCD_INCLUDE_DIR}")
    set(CCD_LIBRARIES "${CCD_LIBRARY}" "${M_LIBRARY}")
    set(ccd_FOUND ON)

    mark_as_advanced(CCD_INCLUDE_DIR CCD_LIBRARY)
  endif()
endif()

#####################################

# Additional include folders
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(cyl_test
  src/cylinder_test.cpp
)

target_compile_options(cyl_test PUBLIC -std=c++11 -Wall -O3)

add_executable(annotateRoadmap
  src/cylinderAnnotation.cpp
  src/convex_collision.cpp
)


target_link_libraries(annotateRoadmap
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${CCD_LIBRARIES}
)

target_compile_options(annotateRoadmap PUBLIC -std=c++11 -Wall)# -fopenmp -O3)




