# Reproducing VOT 2015
To compile use:

mkdir build
cd build
cmake -DVOT2015=ON ..
cd ../matlab/vot-tookit/antrack

// Start matlab and run
run_experiments

# Dependencies
Logging library
https://github.com/google/glog
