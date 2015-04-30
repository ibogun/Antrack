cmake_minimum_required(VERSION 2.8.4)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(test "Build all tests." OFF) # Makes boolean 'test' available.
option(VOT2014 "Build VOT2014-friendly" OFF)
option(experiments "Build all experiments." OFF) # Makes boolean 'test' available.

option(python "Build python wrappers." OFF)







option(parallel "Build parallel dataset evaluation." OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY bin)
# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
set(PROJECT_NAME Robust_tracking_by_detection)
project(${PROJECT_NAME})

#set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED )
find_package(Armadillo REQUIRED)
find_package (Threads REQUIRED)



# If you want your own include/ directory, set this, and then you can do
# include_directories(${COMMON_INCLUDES}) in other CMakeLists.txt files.
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)

################################
# Normal Libraries & Executables
################################
#add_library(project1_lib src/Tests/project1.cpp)
#add_executable(Testing src/Tests/test_project1.cpp)
# Key idea: SEPARATE OUT your main() function into its own file so it can be its
# own executable. Separating out main() means you can add this library to be
# used elsewhere.

set(PYTHON_FILES
	src/python_wrappers/objectness_topy.cpp
)


FILE(GLOB TEST_FILES
	src/Tests/*.cpp
	src/Tests/*.h
	src/Tests/main.cpp
	)

set(FILTER_PRETRAINING_MAIN
	src/main_filter_pretraining.cpp
)

set(PARALLEL_MAIN
  src/main_parallel.cpp
)

set(PROJECT_MAIN
	src/main.cpp
)

set(Experiment_01
	src/Experiments/Experiment_efficient_int_kernel.cpp
	src/Experiments/Experiment_efficient_int_kernel.h
)

set(Superpixels
   src/Superpixels/helper.h
   src/Superpixels/seeds2.h
   src/Superpixels/seeds2.cpp
   src/Superpixels/Plot.h
   src/Superpixels/Plot.cpp
   src/Superpixels/SuperPixels.h
   src/Superpixels/SuperPixels.cpp
   src/Superpixels/Objectness.h
   src/Superpixels/Objectness.cpp
   )


FILE(GLOB TRAX_FILES
	src/Datasets/vot2014_trax/*.h
	src/Datasets/vot2014_trax/*.c
	src/Datasets/vot2014_trax/main.cpp
)



FILE(GLOB SOURCE_FILES
	src/Datasets/*.cpp
	src/Datasets/*.h
	src/Features/*.cpp
	src/Features/*.h
	src/Kernels/*.cpp
	src/Kernels/*.h
	src/Filter/*.cpp
	src/Filter/*.h
	src/Tracker/*.cpp
	src/Tracker/*.h
)



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
  add_executable(RunUnitTests ${SOURCE_FILES} ${TEST_FILES} ${Superpixels})
  #add_executable(RunUnitTests ${SOURCE_FILES})
  # Standard linking to gtest stuff.
  target_link_libraries(RunUnitTests gtest gtest_main ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

  # Extra linking for the project.
  #target_link_libraries(runUnitTests project1_lib)

  # This is so you can do 'make test' to see all your tests run, instead of
  # manually running the executable runUnitTests to see those specific tests.
  #add_test(NAME that-test-I-made COMMAND RunUnitTests)

  # You can also omit NAME and COMMAND. The second argument could be some other
  # test executable.
  #add_test(that-other-test-I-made RunUnitTests)
endif()


if (experiments)
	add_executable(experiment_01_fast_intersection_kernels ${SOURCE_FILES} ${Experiment_01})
	target_link_libraries(experiment_01_fast_intersection_kernels ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

if(VOT2014)
	find_package( Boost COMPONENTS program_options filesystem system  REQUIRED )

	include_directories( ${Boost_INCLUDE_DIR} )
	add_executable(struck_vot2014 ${Superpixels} ${SOURCE_FILES} ${TRAX_FILES})
	target_link_libraries(struck_vot2014 ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
endif()





if(python)
	set(PYTHON_DESTINATION "../python/Experiments/")



	find_package(PythonLibs)

	set(PYTHON_INCLUDE_DIRS /usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/Headers)

	set(PYTHON_LIBRARIES /usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib)

	find_package( Boost COMPONENTS python)
	include_directories (${Boost_INCLUDE_DIRS})

	PYTHON_ADD_MODULE(objectness ${SOURCE_FILES} ${PYTHON_FILES} ${Superpixels})
target_link_libraries(objectness
    boost_python
    ${PYTHON_LIBRARIES}
    ${Boost_LIBRARIES}
	${OpenCV_LIBS}
	${ARMADILLO_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	)
	set_target_properties(objectness PROPERTIES
				LIBRARY_OUTPUT_DIRECTORY ${PYTHON_DESTINATION})

	#FILE(COPY objectness.so DESTINATION ${PYTHON_DESTINATION})
endif()



find_package( Boost COMPONENTS filesystem system REQUIRED )
add_executable(robust_struck_tracker_v1.0 ${Superpixels} ${SOURCE_FILES} ${PROJECT_MAIN})



include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories (${Boost_INCLUDE_DIRS})
target_link_libraries(robust_struck_tracker_v1.0 ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES} )


if (parallel)


	find_package( Boost COMPONENTS program_options filesystem system  REQUIRED )

	include_directories( ${Boost_INCLUDE_DIR} )

  	#add_executable(parallel_main ${Superpixels} ${SOURCE_FILES} ${PARALLEL_MAIN})
	#target_link_libraries(parallel_main ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

	add_executable(struck_filter_experiments ${Superpixels} ${SOURCE_FILES} ${FILTER_PRETRAINING_MAIN})
	target_link_libraries(struck_filter_experiments ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()
