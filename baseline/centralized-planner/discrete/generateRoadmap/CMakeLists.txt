cmake_minimum_required(VERSION 2.8)

project(generateRoadmap)

# set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options filesystem REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
#FIND_PACKAGE(fcl REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)
FIND_PACKAGE(assimp REQUIRED)
FIND_PACKAGE(OMPL REQUIRED)
FIND_PACKAGE(octomap REQUIRED)

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

######################################

# Additional include folders
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}
  ${OMPL_INCLUDE_DIRS}
  ../libCommon/include
  # externalDependencies/ompl_bolt/src
)

add_executable(generateRoadmap
  src/main.cpp
  src/fclHelper.cpp
  # externalDependencies/ompl_bolt/src/ompl/tools/bolt/src/SparseGenerator.cpp
)

message(STATUS ${fcl_LIBRARIES})

link_directories(
  ${FCL_LIBRARY_DIRS}
)

target_link_libraries(generateRoadmap
  ${Boost_LIBRARIES}
  ${fcl_LIBRARIES}
  ${CCD_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${ASSIMP_LIBRARIES}
  ${OMPL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  fcl
  libCommon
)

target_compile_options(generateRoadmap PUBLIC -std=c++11 -Wall)

##

# add_executable(generateRoadmapOMPL
#   src/mainOMPL.cpp
# )

# target_link_libraries(generateRoadmapOMPL
#   ${Boost_LIBRARIES}
#   ${fcl_LIBRARIES}
#   ${CCD_LIBRARIES}
#   ${YAML_CPP_LIBRARIES}
#   ${ASSIMP_LIBRARIES}
#   ${OMPL_LIBRARIES}
#   fcl
# )

# target_compile_options(generateRoadmapOMPL PUBLIC -std=c++11 -Wall)
