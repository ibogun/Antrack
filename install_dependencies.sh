if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get update && sudo apt-get upgrade
    sudo apt-get install libopencv-dev
    sudo apt-get install liblapack-dev
    sudo apt-get install libblas-dev
    sudo apt-get install libboost-dev
    sudo apt-get install libarmadillo-dev
    sudo apt-get install libboost-all-dev
    sudo apt-get install libgflags-dev libgoogle-glog-dev
    # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew update && brew upgrade && brew tap homebrew/science && \
        brew install opencv && brew install armadillo && brew install boost \
                                                              brew install glog

    # Mac OSX
fi
