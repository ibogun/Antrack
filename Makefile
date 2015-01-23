# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Applications/CMake 2.8-12.app/Contents/bin/cmake"

# The command to remove a file.
RM = "/Applications/CMake 2.8-12.app/Contents/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = "/Applications/CMake 2.8-12.app/Contents/bin/ccmake"

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Ivan/Code/Tracking/Antrack

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Ivan/Code/Tracking/Antrack

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	"/Applications/CMake 2.8-12.app/Contents/bin/ccmake" -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	"/Applications/CMake 2.8-12.app/Contents/bin/cmake" -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/Ivan/Code/Tracking/Antrack/CMakeFiles /Users/Ivan/Code/Tracking/Antrack/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/Ivan/Code/Tracking/Antrack/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named robust_struck_tracker_v1.0

# Build rule for target.
robust_struck_tracker_v1.0: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 robust_struck_tracker_v1.0
.PHONY : robust_struck_tracker_v1.0

# fast build rule for target.
robust_struck_tracker_v1.0/fast:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/build
.PHONY : robust_struck_tracker_v1.0/fast

src/Datasets/DataSetWu2013.o: src/Datasets/DataSetWu2013.cpp.o
.PHONY : src/Datasets/DataSetWu2013.o

# target to build an object file
src/Datasets/DataSetWu2013.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/DataSetWu2013.cpp.o
.PHONY : src/Datasets/DataSetWu2013.cpp.o

src/Datasets/DataSetWu2013.i: src/Datasets/DataSetWu2013.cpp.i
.PHONY : src/Datasets/DataSetWu2013.i

# target to preprocess a source file
src/Datasets/DataSetWu2013.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/DataSetWu2013.cpp.i
.PHONY : src/Datasets/DataSetWu2013.cpp.i

src/Datasets/DataSetWu2013.s: src/Datasets/DataSetWu2013.cpp.s
.PHONY : src/Datasets/DataSetWu2013.s

# target to generate assembly for a file
src/Datasets/DataSetWu2013.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/DataSetWu2013.cpp.s
.PHONY : src/Datasets/DataSetWu2013.cpp.s

src/Datasets/Dataset.o: src/Datasets/Dataset.cpp.o
.PHONY : src/Datasets/Dataset.o

# target to build an object file
src/Datasets/Dataset.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/Dataset.cpp.o
.PHONY : src/Datasets/Dataset.cpp.o

src/Datasets/Dataset.i: src/Datasets/Dataset.cpp.i
.PHONY : src/Datasets/Dataset.i

# target to preprocess a source file
src/Datasets/Dataset.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/Dataset.cpp.i
.PHONY : src/Datasets/Dataset.cpp.i

src/Datasets/Dataset.s: src/Datasets/Dataset.cpp.s
.PHONY : src/Datasets/Dataset.s

# target to generate assembly for a file
src/Datasets/Dataset.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Datasets/Dataset.cpp.s
.PHONY : src/Datasets/Dataset.cpp.s

src/Features/Haar.o: src/Features/Haar.cpp.o
.PHONY : src/Features/Haar.o

# target to build an object file
src/Features/Haar.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Haar.cpp.o
.PHONY : src/Features/Haar.cpp.o

src/Features/Haar.i: src/Features/Haar.cpp.i
.PHONY : src/Features/Haar.i

# target to preprocess a source file
src/Features/Haar.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Haar.cpp.i
.PHONY : src/Features/Haar.cpp.i

src/Features/Haar.s: src/Features/Haar.cpp.s
.PHONY : src/Features/Haar.s

# target to generate assembly for a file
src/Features/Haar.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Haar.cpp.s
.PHONY : src/Features/Haar.cpp.s

src/Features/HaarFeature.o: src/Features/HaarFeature.cpp.o
.PHONY : src/Features/HaarFeature.o

# target to build an object file
src/Features/HaarFeature.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeature.cpp.o
.PHONY : src/Features/HaarFeature.cpp.o

src/Features/HaarFeature.i: src/Features/HaarFeature.cpp.i
.PHONY : src/Features/HaarFeature.i

# target to preprocess a source file
src/Features/HaarFeature.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeature.cpp.i
.PHONY : src/Features/HaarFeature.cpp.i

src/Features/HaarFeature.s: src/Features/HaarFeature.cpp.s
.PHONY : src/Features/HaarFeature.s

# target to generate assembly for a file
src/Features/HaarFeature.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeature.cpp.s
.PHONY : src/Features/HaarFeature.cpp.s

src/Features/HaarFeatureSet.o: src/Features/HaarFeatureSet.cpp.o
.PHONY : src/Features/HaarFeatureSet.o

# target to build an object file
src/Features/HaarFeatureSet.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeatureSet.cpp.o
.PHONY : src/Features/HaarFeatureSet.cpp.o

src/Features/HaarFeatureSet.i: src/Features/HaarFeatureSet.cpp.i
.PHONY : src/Features/HaarFeatureSet.i

# target to preprocess a source file
src/Features/HaarFeatureSet.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeatureSet.cpp.i
.PHONY : src/Features/HaarFeatureSet.cpp.i

src/Features/HaarFeatureSet.s: src/Features/HaarFeatureSet.cpp.s
.PHONY : src/Features/HaarFeatureSet.s

# target to generate assembly for a file
src/Features/HaarFeatureSet.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/HaarFeatureSet.cpp.s
.PHONY : src/Features/HaarFeatureSet.cpp.s

src/Features/Histogram.o: src/Features/Histogram.cpp.o
.PHONY : src/Features/Histogram.o

# target to build an object file
src/Features/Histogram.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Histogram.cpp.o
.PHONY : src/Features/Histogram.cpp.o

src/Features/Histogram.i: src/Features/Histogram.cpp.i
.PHONY : src/Features/Histogram.i

# target to preprocess a source file
src/Features/Histogram.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Histogram.cpp.i
.PHONY : src/Features/Histogram.cpp.i

src/Features/Histogram.s: src/Features/Histogram.cpp.s
.PHONY : src/Features/Histogram.s

# target to generate assembly for a file
src/Features/Histogram.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/Histogram.cpp.s
.PHONY : src/Features/Histogram.cpp.s

src/Features/RawFeatures.o: src/Features/RawFeatures.cpp.o
.PHONY : src/Features/RawFeatures.o

# target to build an object file
src/Features/RawFeatures.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/RawFeatures.cpp.o
.PHONY : src/Features/RawFeatures.cpp.o

src/Features/RawFeatures.i: src/Features/RawFeatures.cpp.i
.PHONY : src/Features/RawFeatures.i

# target to preprocess a source file
src/Features/RawFeatures.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/RawFeatures.cpp.i
.PHONY : src/Features/RawFeatures.cpp.i

src/Features/RawFeatures.s: src/Features/RawFeatures.cpp.s
.PHONY : src/Features/RawFeatures.s

# target to generate assembly for a file
src/Features/RawFeatures.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Features/RawFeatures.cpp.s
.PHONY : src/Features/RawFeatures.cpp.s

src/Filter/KalmanFilter.o: src/Filter/KalmanFilter.cpp.o
.PHONY : src/Filter/KalmanFilter.o

# target to build an object file
src/Filter/KalmanFilter.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilter.cpp.o
.PHONY : src/Filter/KalmanFilter.cpp.o

src/Filter/KalmanFilter.i: src/Filter/KalmanFilter.cpp.i
.PHONY : src/Filter/KalmanFilter.i

# target to preprocess a source file
src/Filter/KalmanFilter.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilter.cpp.i
.PHONY : src/Filter/KalmanFilter.cpp.i

src/Filter/KalmanFilter.s: src/Filter/KalmanFilter.cpp.s
.PHONY : src/Filter/KalmanFilter.s

# target to generate assembly for a file
src/Filter/KalmanFilter.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilter.cpp.s
.PHONY : src/Filter/KalmanFilter.cpp.s

src/Filter/KalmanFilterGenerator.o: src/Filter/KalmanFilterGenerator.cpp.o
.PHONY : src/Filter/KalmanFilterGenerator.o

# target to build an object file
src/Filter/KalmanFilterGenerator.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilterGenerator.cpp.o
.PHONY : src/Filter/KalmanFilterGenerator.cpp.o

src/Filter/KalmanFilterGenerator.i: src/Filter/KalmanFilterGenerator.cpp.i
.PHONY : src/Filter/KalmanFilterGenerator.i

# target to preprocess a source file
src/Filter/KalmanFilterGenerator.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilterGenerator.cpp.i
.PHONY : src/Filter/KalmanFilterGenerator.cpp.i

src/Filter/KalmanFilterGenerator.s: src/Filter/KalmanFilterGenerator.cpp.s
.PHONY : src/Filter/KalmanFilterGenerator.s

# target to generate assembly for a file
src/Filter/KalmanFilterGenerator.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Filter/KalmanFilterGenerator.cpp.s
.PHONY : src/Filter/KalmanFilterGenerator.cpp.s

src/Kernels/ApproximateKernel.o: src/Kernels/ApproximateKernel.cpp.o
.PHONY : src/Kernels/ApproximateKernel.o

# target to build an object file
src/Kernels/ApproximateKernel.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/ApproximateKernel.cpp.o
.PHONY : src/Kernels/ApproximateKernel.cpp.o

src/Kernels/ApproximateKernel.i: src/Kernels/ApproximateKernel.cpp.i
.PHONY : src/Kernels/ApproximateKernel.i

# target to preprocess a source file
src/Kernels/ApproximateKernel.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/ApproximateKernel.cpp.i
.PHONY : src/Kernels/ApproximateKernel.cpp.i

src/Kernels/ApproximateKernel.s: src/Kernels/ApproximateKernel.cpp.s
.PHONY : src/Kernels/ApproximateKernel.s

# target to generate assembly for a file
src/Kernels/ApproximateKernel.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/ApproximateKernel.cpp.s
.PHONY : src/Kernels/ApproximateKernel.cpp.s

src/Kernels/IntersectionKernel.o: src/Kernels/IntersectionKernel.cpp.o
.PHONY : src/Kernels/IntersectionKernel.o

# target to build an object file
src/Kernels/IntersectionKernel.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel.cpp.o
.PHONY : src/Kernels/IntersectionKernel.cpp.o

src/Kernels/IntersectionKernel.i: src/Kernels/IntersectionKernel.cpp.i
.PHONY : src/Kernels/IntersectionKernel.i

# target to preprocess a source file
src/Kernels/IntersectionKernel.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel.cpp.i
.PHONY : src/Kernels/IntersectionKernel.cpp.i

src/Kernels/IntersectionKernel.s: src/Kernels/IntersectionKernel.cpp.s
.PHONY : src/Kernels/IntersectionKernel.s

# target to generate assembly for a file
src/Kernels/IntersectionKernel.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel.cpp.s
.PHONY : src/Kernels/IntersectionKernel.cpp.s

src/Kernels/IntersectionKernel_fast.o: src/Kernels/IntersectionKernel_fast.cpp.o
.PHONY : src/Kernels/IntersectionKernel_fast.o

# target to build an object file
src/Kernels/IntersectionKernel_fast.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel_fast.cpp.o
.PHONY : src/Kernels/IntersectionKernel_fast.cpp.o

src/Kernels/IntersectionKernel_fast.i: src/Kernels/IntersectionKernel_fast.cpp.i
.PHONY : src/Kernels/IntersectionKernel_fast.i

# target to preprocess a source file
src/Kernels/IntersectionKernel_fast.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel_fast.cpp.i
.PHONY : src/Kernels/IntersectionKernel_fast.cpp.i

src/Kernels/IntersectionKernel_fast.s: src/Kernels/IntersectionKernel_fast.cpp.s
.PHONY : src/Kernels/IntersectionKernel_fast.s

# target to generate assembly for a file
src/Kernels/IntersectionKernel_fast.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/IntersectionKernel_fast.cpp.s
.PHONY : src/Kernels/IntersectionKernel_fast.cpp.s

src/Kernels/Kernel.o: src/Kernels/Kernel.cpp.o
.PHONY : src/Kernels/Kernel.o

# target to build an object file
src/Kernels/Kernel.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Kernel.cpp.o
.PHONY : src/Kernels/Kernel.cpp.o

src/Kernels/Kernel.i: src/Kernels/Kernel.cpp.i
.PHONY : src/Kernels/Kernel.i

# target to preprocess a source file
src/Kernels/Kernel.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Kernel.cpp.i
.PHONY : src/Kernels/Kernel.cpp.i

src/Kernels/Kernel.s: src/Kernels/Kernel.cpp.s
.PHONY : src/Kernels/Kernel.s

# target to generate assembly for a file
src/Kernels/Kernel.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Kernel.cpp.s
.PHONY : src/Kernels/Kernel.cpp.s

src/Kernels/RBFKernel.o: src/Kernels/RBFKernel.cpp.o
.PHONY : src/Kernels/RBFKernel.o

# target to build an object file
src/Kernels/RBFKernel.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/RBFKernel.cpp.o
.PHONY : src/Kernels/RBFKernel.cpp.o

src/Kernels/RBFKernel.i: src/Kernels/RBFKernel.cpp.i
.PHONY : src/Kernels/RBFKernel.i

# target to preprocess a source file
src/Kernels/RBFKernel.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/RBFKernel.cpp.i
.PHONY : src/Kernels/RBFKernel.cpp.i

src/Kernels/RBFKernel.s: src/Kernels/RBFKernel.cpp.s
.PHONY : src/Kernels/RBFKernel.s

# target to generate assembly for a file
src/Kernels/RBFKernel.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/RBFKernel.cpp.s
.PHONY : src/Kernels/RBFKernel.cpp.s

src/Kernels/Spline.o: src/Kernels/Spline.cpp.o
.PHONY : src/Kernels/Spline.o

# target to build an object file
src/Kernels/Spline.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Spline.cpp.o
.PHONY : src/Kernels/Spline.cpp.o

src/Kernels/Spline.i: src/Kernels/Spline.cpp.i
.PHONY : src/Kernels/Spline.i

# target to preprocess a source file
src/Kernels/Spline.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Spline.cpp.i
.PHONY : src/Kernels/Spline.cpp.i

src/Kernels/Spline.s: src/Kernels/Spline.cpp.s
.PHONY : src/Kernels/Spline.s

# target to generate assembly for a file
src/Kernels/Spline.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Kernels/Spline.cpp.s
.PHONY : src/Kernels/Spline.cpp.s

src/Tracker/LocationSampler.o: src/Tracker/LocationSampler.cpp.o
.PHONY : src/Tracker/LocationSampler.o

# target to build an object file
src/Tracker/LocationSampler.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/LocationSampler.cpp.o
.PHONY : src/Tracker/LocationSampler.cpp.o

src/Tracker/LocationSampler.i: src/Tracker/LocationSampler.cpp.i
.PHONY : src/Tracker/LocationSampler.i

# target to preprocess a source file
src/Tracker/LocationSampler.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/LocationSampler.cpp.i
.PHONY : src/Tracker/LocationSampler.cpp.i

src/Tracker/LocationSampler.s: src/Tracker/LocationSampler.cpp.s
.PHONY : src/Tracker/LocationSampler.s

# target to generate assembly for a file
src/Tracker/LocationSampler.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/LocationSampler.cpp.s
.PHONY : src/Tracker/LocationSampler.cpp.s

src/Tracker/OLaRank_old.o: src/Tracker/OLaRank_old.cpp.o
.PHONY : src/Tracker/OLaRank_old.o

# target to build an object file
src/Tracker/OLaRank_old.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/OLaRank_old.cpp.o
.PHONY : src/Tracker/OLaRank_old.cpp.o

src/Tracker/OLaRank_old.i: src/Tracker/OLaRank_old.cpp.i
.PHONY : src/Tracker/OLaRank_old.i

# target to preprocess a source file
src/Tracker/OLaRank_old.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/OLaRank_old.cpp.i
.PHONY : src/Tracker/OLaRank_old.cpp.i

src/Tracker/OLaRank_old.s: src/Tracker/OLaRank_old.cpp.s
.PHONY : src/Tracker/OLaRank_old.s

# target to generate assembly for a file
src/Tracker/OLaRank_old.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/OLaRank_old.cpp.s
.PHONY : src/Tracker/OLaRank_old.cpp.s

src/Tracker/Struck.o: src/Tracker/Struck.cpp.o
.PHONY : src/Tracker/Struck.o

# target to build an object file
src/Tracker/Struck.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/Struck.cpp.o
.PHONY : src/Tracker/Struck.cpp.o

src/Tracker/Struck.i: src/Tracker/Struck.cpp.i
.PHONY : src/Tracker/Struck.i

# target to preprocess a source file
src/Tracker/Struck.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/Struck.cpp.i
.PHONY : src/Tracker/Struck.cpp.i

src/Tracker/Struck.s: src/Tracker/Struck.cpp.s
.PHONY : src/Tracker/Struck.s

# target to generate assembly for a file
src/Tracker/Struck.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/Struck.cpp.s
.PHONY : src/Tracker/Struck.cpp.s

src/Tracker/SupportVector.o: src/Tracker/SupportVector.cpp.o
.PHONY : src/Tracker/SupportVector.o

# target to build an object file
src/Tracker/SupportVector.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/SupportVector.cpp.o
.PHONY : src/Tracker/SupportVector.cpp.o

src/Tracker/SupportVector.i: src/Tracker/SupportVector.cpp.i
.PHONY : src/Tracker/SupportVector.i

# target to preprocess a source file
src/Tracker/SupportVector.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/SupportVector.cpp.i
.PHONY : src/Tracker/SupportVector.cpp.i

src/Tracker/SupportVector.s: src/Tracker/SupportVector.cpp.s
.PHONY : src/Tracker/SupportVector.s

# target to generate assembly for a file
src/Tracker/SupportVector.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/Tracker/SupportVector.cpp.s
.PHONY : src/Tracker/SupportVector.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/robust_struck_tracker_v1.0.dir/build.make CMakeFiles/robust_struck_tracker_v1.0.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... robust_struck_tracker_v1.0"
	@echo "... src/Datasets/DataSetWu2013.o"
	@echo "... src/Datasets/DataSetWu2013.i"
	@echo "... src/Datasets/DataSetWu2013.s"
	@echo "... src/Datasets/Dataset.o"
	@echo "... src/Datasets/Dataset.i"
	@echo "... src/Datasets/Dataset.s"
	@echo "... src/Features/Haar.o"
	@echo "... src/Features/Haar.i"
	@echo "... src/Features/Haar.s"
	@echo "... src/Features/HaarFeature.o"
	@echo "... src/Features/HaarFeature.i"
	@echo "... src/Features/HaarFeature.s"
	@echo "... src/Features/HaarFeatureSet.o"
	@echo "... src/Features/HaarFeatureSet.i"
	@echo "... src/Features/HaarFeatureSet.s"
	@echo "... src/Features/Histogram.o"
	@echo "... src/Features/Histogram.i"
	@echo "... src/Features/Histogram.s"
	@echo "... src/Features/RawFeatures.o"
	@echo "... src/Features/RawFeatures.i"
	@echo "... src/Features/RawFeatures.s"
	@echo "... src/Filter/KalmanFilter.o"
	@echo "... src/Filter/KalmanFilter.i"
	@echo "... src/Filter/KalmanFilter.s"
	@echo "... src/Filter/KalmanFilterGenerator.o"
	@echo "... src/Filter/KalmanFilterGenerator.i"
	@echo "... src/Filter/KalmanFilterGenerator.s"
	@echo "... src/Kernels/ApproximateKernel.o"
	@echo "... src/Kernels/ApproximateKernel.i"
	@echo "... src/Kernels/ApproximateKernel.s"
	@echo "... src/Kernels/IntersectionKernel.o"
	@echo "... src/Kernels/IntersectionKernel.i"
	@echo "... src/Kernels/IntersectionKernel.s"
	@echo "... src/Kernels/IntersectionKernel_fast.o"
	@echo "... src/Kernels/IntersectionKernel_fast.i"
	@echo "... src/Kernels/IntersectionKernel_fast.s"
	@echo "... src/Kernels/Kernel.o"
	@echo "... src/Kernels/Kernel.i"
	@echo "... src/Kernels/Kernel.s"
	@echo "... src/Kernels/RBFKernel.o"
	@echo "... src/Kernels/RBFKernel.i"
	@echo "... src/Kernels/RBFKernel.s"
	@echo "... src/Kernels/Spline.o"
	@echo "... src/Kernels/Spline.i"
	@echo "... src/Kernels/Spline.s"
	@echo "... src/Tracker/LocationSampler.o"
	@echo "... src/Tracker/LocationSampler.i"
	@echo "... src/Tracker/LocationSampler.s"
	@echo "... src/Tracker/OLaRank_old.o"
	@echo "... src/Tracker/OLaRank_old.i"
	@echo "... src/Tracker/OLaRank_old.s"
	@echo "... src/Tracker/Struck.o"
	@echo "... src/Tracker/Struck.i"
	@echo "... src/Tracker/Struck.s"
	@echo "... src/Tracker/SupportVector.o"
	@echo "... src/Tracker/SupportVector.i"
	@echo "... src/Tracker/SupportVector.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

