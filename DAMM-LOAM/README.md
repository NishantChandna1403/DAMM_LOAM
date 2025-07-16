# Fusion Ws
```
git clone -b fusion https://github.com/NishantChandna1403/Fusion_ws.git
```
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them.

* Docker
```
# Install Docker using convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh

# Post-install configuration
sudo groupadd docker
sudo usermod -aG docker $USER
sudo systemctl enable docker.service
sudo systemctl enable containerd.service

# Verify installation
sudo systemctl is-enabled docker

```
### Installing and Running

This repository includes a `.devcontainer` configuration. To set up the development environment:

1. Open the repository in VS Code.
2. When prompted, select **Reopen in Container**. If not prompted, use the **Command Palette** (Ctrl+Shift+P) and select **Remote-Containers: Reopen in Container**.
3. VS Code will automatically build the Docker image and set up the development environment for you.
4. Once the container is built and opened, you can access the development environment directly.

### Running the Odometry Fuser
To start the fuser and get output on topics `/odometry_fusion_node/fused_odometry` and `/odometry_fusion_node/fused_path`, use the following command:
```
rosrun odometry_fuser odometry_fuser_node
```

### Launching FastLIO
To launch FastLIO with the `mapping_avia.launch` file, use:
```
roslaunch fast_lio mapping_avia.launch
```

### Launching VINS Mono
To launch VINS Mono with the `realsense_color.launch` file, use:
```
roslaunch vins_estimator realsense_color.launch
```

### Playing a Rosbag
To play a specific rosbag file, for example `2025-06-10-17-27-17.bag`, use:
```
rosbag play 2025-06-10-17-27-17.bag
```
# DAMM-LOAM
