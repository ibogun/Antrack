# Pull base image.
FROM ubuntu:14.04.4

# Install.
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y git && \
  rm -rf /var/lib/apt/lists/*

ADD install_dependencies.sh /
RUN chmod +x install_dependencies.sh
RUN /install_dependencies.sh

RUN \
   mkdir build && cd build && export PROJECT_HOME=$(pwd) && cd ${PROJECT_HOME} && \
   cmake -Dtest=on -DDeepFeatures=OFF .. && \
   make && ./bin/RunUnitTests

# Add files.
# # ADD root/.bashrc /root/.bashrc
# # ADD root/.gitconfig /root/.gitconfig
# # ADD root/.scripts /root/.scripts

# Set environment variables.
# ENV HOME /root

# Define working directory.
# WORKDIR /root

# Define default command.
CMD ["bash"]