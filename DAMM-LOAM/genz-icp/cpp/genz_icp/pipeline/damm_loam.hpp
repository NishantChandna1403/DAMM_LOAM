// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill Stachniss.
// Modified by Daehan Lee, Hyungtae Lim, and Soohee Han, 2024
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "damm_loam/core/Deskew.hpp"
#include "damm_loam/core/Threshold.hpp"
#include "damm_loam/core/VoxelHashMap.hpp"
#include "damm_loam/core/Registration.hpp"

namespace damm_loam::pipeline {

struct damm-Config {
    // map params
    double max_range = 100.0;
    double min_range = 0.5;
    double map_cleanup_radius = 400.0;
    int max_points_per_voxel = 1;

    // voxelize params
    double voxel_size = 0.25;
    int desired_num_voxelized_points = 2000;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;
    double planarity_threshold = 0.1;

    // Motion compensation
    bool deskew = false;

    // registration params
    int max_num_iterations = 150;
    double convergence_criterion = 0.0001;
};

class damm_loam {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit damm_loam(const damm-Config &config)
        : config_(config),
          registration_(config.max_num_iterations, config.convergence_criterion),
          local_map_(config.voxel_size, config.max_range, config.map_cleanup_radius, config.planarity_threshold, config.max_points_per_voxel),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

    damm_loam() : damm_loam(damm-Config{}) {}

public:
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame);
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                      const std::vector<double> &timestamps);
    
    // Methods for 5 separate point cloud types
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &ground_points,
                                      const std::vector<Eigen::Vector3d> &roof_points,
                                      const std::vector<Eigen::Vector3d> &wall_points,
                                      const std::vector<Eigen::Vector3d> &edge_points,
                                      const std::vector<Eigen::Vector3d> &non_planar_points);
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &ground_points,
                                      const std::vector<Eigen::Vector3d> &roof_points,
                                      const std::vector<Eigen::Vector3d> &wall_points,
                                      const std::vector<Eigen::Vector3d> &edge_points,
                                      const std::vector<Eigen::Vector3d> &non_planar_points,
                                      const std::vector<double> &timestamps);
    
    Vector3dVectorTuple Voxelize(const std::vector<Eigen::Vector3d> &frame, double voxel_size) const;
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();

public:
    // Extra C++ API to facilitate ROS debugging
    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };
    std::vector<Sophus::SE3d> poses() const { return poses_; };

private:
    // damm-loam pipeline modules
    std::vector<Sophus::SE3d> poses_;
    damm-Config config_;
    Registration registration_;
    VoxelHashMap local_map_;
    AdaptiveThreshold adaptive_threshold_;
};

}  // namespace damm_loam::pipeline
