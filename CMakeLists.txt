cmake_minimum_required(VERSION 2.8.4)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(test "Build all tests." OFF) # Makes boolean 'test' available.

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
set(PROJECT_NAME Robust_tracking_by_detection)
project(${PROJECT_NAME})

#set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED )
find_package(Armadillo REQUIRED)
# If you want your own include/ directory, set this, and then you can do
# include_directories(${COMMON_INCLUDES}) in other CMakeLists.txt files.
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)

################################
# Normal Libraries & Executables
################################
add_library(project1_lib src/Tests/project1.cpp)
add_executable(Testing src/Tests/main.cpp)
# Key idea: SEPARATE OUT your main() function into its own file so it can be its
# own executable. Separating out main() means you can add this library to be
# used elsewhere.


set(SOURCE_FILES
    src/Datasets/Dataset.cpp
    src/Datasets/Dataset.h
    src/Datasets/DataSetWu2013.cpp
    src/Datasets/DataSetWu2013.h
    src/Features/Feature.h
    src/Features/Haar.cpp
    src/Features/Haar.h
    src/Features/HaarFeature.cpp
    src/Features/HaarFeature.h
    src/Features/HaarFeatureSet.cpp
    src/Features/HaarFeatureSet.h
    src/Features/Histogram.cpp
    src/Features/Histogram.h
    src/Kernels/IntersectionKernel.cpp
    src/Kernels/IntersectionKernel.h
    src/Filter/KalmanFilter.cpp
    src/Filter/KalmanFilter.h
    src/Filter/KalmanFilterGenerator.cpp
    src/Filter/KalmanFilterGenerator.h
    src/Kernels/Kernel.cpp
    src/Kernels/Kernel.h
    src/Tracker/LocationSampler.cpp
    src/Tracker/LocationSampler.h
    src/main.cpp
    src/Tracker/OLaRank_old.cpp
    src/Tracker/OLaRank_old.h
    src/Features/RawFeatures.cpp
    src/Features/RawFeatures.h
    src/Kernels/RBFKernel.cpp
    src/Kernels/RBFKernel.h
    src/Tracker/Struck.cpp
    src/Tracker/Struck.h
    src/Tracker/SupportVector.cpp
    src/Tracker/SupportVector.h
	src/Kernels/IntersectionKernel_fast.cpp
	src/Kernels/IntersectionKernel_fast.h
    Makefile)

target_link_libraries(Testing project1_lib)

################################
# Testing
################################
if (test)
  # This adds another subdirectory, which has 'project(gtest)'.
  add_subdirectory(lib/gtest-1.6.0)

  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  # 'project(gtest)' above.
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  ##############
  # Unit Tests
  ##############
  add_executable(RunUnitTests src/Tests/test_project1.cpp)

  # Standard linking to gtest stuff.
  target_link_libraries(RunUnitTests gtest gtest_main)

  # Extra linking for the project.
  #target_link_libraries(runUnitTests project1_lib)

  # This is so you can do 'make test' to see all your tests run, instead of
  # manually running the executable runUnitTests to see those specific tests.
  add_test(NAME that-test-I-made COMMAND RunUnitTests)

  # You can also omit NAME and COMMAND. The second argument could be some other
  # test executable.
  add_test(that-other-test-I-made RunUnitTests)
endif()


add_executable(robust_struck_tracker_v1.0 ${SOURCE_FILES})
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${ARMADILLO_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(robust_struck_tracker_v1.0 ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES})
