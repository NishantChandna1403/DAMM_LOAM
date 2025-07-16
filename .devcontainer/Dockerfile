FROM ghcr.io/soham2560/noetic:latest

# Add metadata labels
LABEL org.opencontainers.image.description="DAMM-LOAM"
LABEL org.opencontainers.image.title="DAMM-LOAM"
LABEL org.opencontainers.image.vendor="Nishant"
LABEL org.opencontainers.image.source="https://github.com/NishantChandna1403/DAMM-LOAM.git"
LABEL maintainer="chandna.nishant@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

# Set default shell
SHELL ["/bin/bash", "-c"]

# Install basic dependencies including PCL
RUN sudo apt-get update && sudo apt-get -y install --no-install-recommends \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    wget \
    cmake \
    build-essential \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    at \
    libaria-dev \
    v4l-utils \
    xterm \
    libpcl-dev \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-pcl-ros \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /home/container_user
