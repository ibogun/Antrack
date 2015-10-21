if [[ "$OSTYPE" == "linux-gnu" ]]; then
    
    set -x
    export DEBIAN_FRONTEND=noninteractive
    add-apt-repository -y ppa:tuleu/precise-backports
    sudo apt-get update && sudo apt-get upgrade -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
    sudo apt-get -qq install libopencv-dev
    sudo apt-get -qq install liblapack-dev
    sudo apt-get -qq install libblas-dev
    sudo apt-get -qq install libboost-dev
    sudo apt-get -qq install libarmadillo-dev
    sudo apt-get -qq install libboost-all-dev
    sudo apt-get -qq install libgflags-dev libgoogle-glog-dev
    # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew update && brew upgrade && brew tap homebrew/science && \
        brew install opencv && brew install armadillo && brew install boost \
                                                              brew install glog

    # Mac OSX
fi
