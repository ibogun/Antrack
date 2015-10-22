if [[ "$OSTYPE" == "linux-gnu" ]]; then
    
    set -x
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update && sudo apt-get upgrade -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
    sudo apt-get -qq install libopencv-dev
    sudo apt-get -qq install liblapack-dev
    sudo apt-get -qq install libblas-dev
    sudo apt-get -qq install libboost-dev
    sudo apt-get -qq install libarmadillo-dev
    sudo apt-get -qq install libboost-all-dev
    # install cmake 3.x
    sudo apt-get install build-essential
    wget http://www.cmake.org/files/v3.2/cmake-3.2.2.tar.gz
    tar xf cmake-3.2.2.tar.gz
    cd cmake-3.2.2./
    ./configure
    make && sudo make install
    cd ..
    
    # install glog
    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    cd glog-0.3.3
    ./configure
    make && make install
    cd ..
    # gflags
    wget https://github.com/schuhschuh/gflags/archive/master.zip
    unzip master.zip
    cd gflags-master
    mkdir build && cd build
    export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
    make && sudo make install
    cd ..
    
    # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew update && brew upgrade && brew tap homebrew/science && \
        brew install opencv && brew install armadillo && brew install boost \
                                                              brew install glog

    # Mac OSX
fi
