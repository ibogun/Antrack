# Pull base image.
FROM ubuntu:14.04.4
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install.
RUN  apt-get update && apt-get install -yq libopencv-dev liblapack-dev libblas-dev libboost-dev libarmadillo-dev libboost-all-dev libgoogle-glog-dev cmake unzip build-essential wget git


# install gflags
RUN wget --no-check-certificate https://github.com/schuhschuh/gflags/archive/master.zip && unzip master.zip
WORKDIR gflags-master
RUN mkdir build && cd build && export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1 &&  make install && cd ../..

ADD ./src/ /Antrack/src/
ADD CMakeLists.txt /Antrack/CMakeLists.txt
ADD ./lib/ /Antrack/lib/

# RUN git clone https://github.com/ibogun/Antrack.git 
# WORKDIR Antrack
RUN cd /Antrack/ && mkdir build && cd build && cmake .. && make

# Define default command.
CMD ["bash"]