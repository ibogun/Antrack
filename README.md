# Antrack [![Build Status](https://travis-ci.org/ibogun/Antrack.svg?branch=master)](https://travis-ci.org/ibogun/Antrack)
Antrack is an open source implementation of the Structured Tracker and tracking evaluation suite. The original structured tracker was introduced by Hare et. al. 2011 and achieved state-of-the-art in the benchmark by Wu et. al 2012. This implementation extends the structured tracker by adding [Robust Kalman](http://my.fit.edu/~ibogun2010/Projects/Robust_tracking_by_detection/index.html) filter and [objectness priors](http://my.fit.edu/~ibogun2010/Projects/Object_aware_tracking/index.html). Each extension is independent of each other and improves tracking metrics on Wu et. al. 2012 dataset.

# Installation
Dependencies installation was tested on Ubuntu 14.04. Mac OS X 10.11 support is experimental.
## Ubuntu 14.04
Script ``install_dependencies.sh`` will install all dependencies automatically. To compile the code it is neccessary to have compiler which
 accepts ``c++11`` flag (gcc > 4.9). To install necessary compiler see [gist](https://gist.github.com/ibogun/ec0a4005c25df57a1b9d).

To compile do:

    ./install_dependencies.sh
## Mac OS X 10.11
 Script ``install_dependencies.sh`` will try to install dependencies using ``homebrew``.

## List of dependencies
* [OpenCV 2.4.11](http://opencv.org/)
* [Armadillo C++](http://arma.sourceforge.net/)
* [Boost](http://www.boost.org/)
* [glog](https://github.com/google/glogg)
* [gflag](https://github.com/gflags/gflags)


# Using
## C++ interface
Coming soon

## Python interface
Coming soon.

## Reproducing VOT 2015
[VOT 2015](http://www.votchallenge.net/vot2015/dataset.html) is a popular dataset for tracking evaluation with it's own evaluatin protocol. To evaluate our tracker on VOT 2015 download evaluation toolkit. Compile the tracker using:

    mkdir build
    cd build
    cmake -DVOT2015=ON ..
    # should create $Antrack/matlab/build/bin/struck_vot2014 binary
    # example of the file to use with VOT2015 is in #Antrack/matlab/tracker_RobStruck.m

For further details see how to integrate the tracker with the tookit [click here](http://www.votchallenge.net/howto/integration.html).
## Reproducing results on Wu et. al. 2013 dataset

### Compiling
    mkdir build
    cd build
    cmake -DCVPR2016=ON ..
    make -j8

### Running

See more examples in the ``scripts/`` folder.

    ./cvpr2016 \        
        --datasetSaveLocation=${datasetSaveLocation}\ --filter=${filter} \ # use robust filter or not RobStruck/Struck
        --nThreads=${nThreads} \ # number of threads to use
        --updateEveryNframes=${updateEveryNFrames} \ # how often the tracker should update
        --b=${b} --P=${P_param} --Q=${Q} --R=${R} \ # Robust Kalman Filter parameters
        --feature=${feature} --kernel=${kernel} \ # Feature-kernel pairs; for best results use hogANDhist & int
        --prefix=${prefix} \ # prefix when saving results of the files
        --lambda_s=${lambda_s} \        # straddling lambda use 0
        --lambda_e=${lambda_e} \ # edge density lambda use 0.4
        --inner=${inner} \ # inner scale parameter; best - 0.9
        --straddeling_threshold=${straddeling_threshold} \ # threshold defining when to cutoff straddling; best: 1.5
        --experiment_type=${experiment_type} \ # 0 if evaluation should be performed on 50 videos, 1 if SRE+TRE should be performed (1632 video runs)
        --tracker_type=${tracker_type}\ # 0 - RobStruck, 1 - ObjStruck
