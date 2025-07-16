// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill Stachniss.
// Modified by Jeevan Lee, Hyungtae Lim, and Soohee Han, 2024
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
#include "damm_loam.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>


#include "damm_loam/core/Deskew.hpp"
#include "damm_loam/core/Preprocessing.hpp"
#include "damm_loam/core/Registration.hpp"
#include "damm_loam/core/VoxelHashMap.hpp"

namespace damm_loam::pipeline {

// RegisterFrame method for 5 separate point cloud types
damm_loam::Vector3dVectorTuple damm_loam::RegisterFrame(const std::vector<Eigen::Vector3d> &ground_points,
                                                    const std::vector<Eigen::Vector3d> &roof_points,
                                                    const std::vector<Eigen::Vector3d> &wall_points,
                                                    const std::vector<Eigen::Vector3d> &edge_points,
                                                    const std::vector<Eigen::Vector3d> &non_planar_points,
                                                    const std::vector<double> &timestamps) {
    // Apply deskewing to each point cloud type individually
    const auto &[deskew_ground, deskew_roof, deskew_wall, deskew_edge, deskew_non_planar] = [&]() -> std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> {
        if (!config_.deskew || timestamps.empty()) return {ground_points, roof_points, wall_points, edge_points, non_planar_points};
        
        // If not enough poses for the estimation, do not de-skew
        const size_t N = poses_.size();
        if (N <= 2) return {ground_points, roof_points, wall_points, edge_points, non_planar_points};

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];

        auto deskew_ground = DeSkewScan(ground_points, timestamps, start_pose, finish_pose);
        auto deskew_roof = DeSkewScan(roof_points, timestamps, start_pose, finish_pose);
        auto deskew_wall = DeSkewScan(wall_points, timestamps, start_pose, finish_pose);
        auto deskew_edge = DeSkewScan(edge_points, timestamps, start_pose, finish_pose);
        auto deskew_non_planar = DeSkewScan(non_planar_points, timestamps, start_pose, finish_pose);
        
        return {deskew_ground, deskew_roof, deskew_wall, deskew_edge, deskew_non_planar};
    }();
    
    // Preprocess each point cloud type individually
    const auto cropped_ground = Preprocess(deskew_ground, config_.max_range, config_.min_range);
    const auto cropped_roof = Preprocess(deskew_roof, config_.max_range, config_.min_range);
    const auto cropped_wall = Preprocess(deskew_wall, config_.max_range, config_.min_range);
    const auto cropped_edge = Preprocess(deskew_edge, config_.max_range, config_.min_range);
    const auto cropped_non_planar = Preprocess(deskew_non_planar, config_.max_range, config_.min_range);

    // Combine all points for adaptive voxel size calculation
    std::vector<Eigen::Vector3d> combined_frame;
    combined_frame.reserve(cropped_ground.size() + cropped_roof.size() + cropped_wall.size() + 
                          cropped_edge.size() + cropped_non_planar.size());
    combined_frame.insert(combined_frame.end(), cropped_ground.begin(), cropped_ground.end());
    combined_frame.insert(combined_frame.end(), cropped_roof.begin(), cropped_roof.end());
    combined_frame.insert(combined_frame.end(), cropped_wall.begin(), cropped_wall.end());
    combined_frame.insert(combined_frame.end(), cropped_edge.begin(), cropped_edge.end());
    combined_frame.insert(combined_frame.end(), cropped_non_planar.begin(), cropped_non_planar.end());

    // Adapt voxel size based on LOCUS 2.0's adaptive voxel grid filter
    static double voxel_size = config_.voxel_size; // Initial voxel size
    const auto source_tmp = VoxelDownsample(combined_frame, voxel_size);
    double adaptive_voxel_size = Clamp(voxel_size * static_cast<double>(source_tmp.size()) / static_cast<double>(config_.desired_num_voxelized_points), 0.02, 2.0);

    // Downsample each point cloud type individually
    const auto downsampled_ground = VoxelDownsample(cropped_ground, std::max(adaptive_voxel_size, 0.02));
    const auto downsampled_roof = VoxelDownsample(cropped_roof, std::max(adaptive_voxel_size, 0.02));
    const auto downsampled_wall = VoxelDownsample(cropped_wall, std::max(adaptive_voxel_size, 0.02));
    const auto downsampled_edge = VoxelDownsample(cropped_edge, std::max(0.7, 0.02));
    const auto downsampled_non_planar = VoxelDownsample(cropped_non_planar, std::max(0.7, 0.02));

    // Voxelize combined frame for local map update
    const auto frame_downsample = VoxelDownsample(combined_frame, std::max(adaptive_voxel_size * 0.5, 0.02));
    voxel_size = adaptive_voxel_size; // Save for the next frame

    // Get motion prediction and adaptive threshold
    const double sigma = GetAdaptiveThreshold();

    // Compute initial_guess for ICP
    const auto prediction = GetPredictionModel();
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    const auto initial_guess = last_pose * prediction;

    // Run damm-loam with individual downsampled point clouds
    const auto &[new_pose, registered_planar, registered_non_planar] = registration_.RegisterFrame(
        downsampled_ground,     // Ground points
        downsampled_roof,       // Roof points
        downsampled_wall,       // Wall points
        downsampled_edge,       // Edge points
        downsampled_non_planar, // Non-planar points
        local_map_,             // Voxel map
        initial_guess,          // Initial guess
        3.0 * sigma,            // Max correspondence distance
        sigma / 3.0);           // Kernel

    // Update adaptive threshold and local map
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    poses_.push_back(new_pose);

    // Return registered planar and non-planar points
    const auto result = std::make_tuple(registered_planar, registered_non_planar);
    
    // Add processed points to individual hashmaps
    if (!poses_.empty()) {
        const auto &latest_pose = poses_.back();
        
        // Transform and add to individual hashmaps using downsampled points
        auto add_to_map = [&](const auto &points, auto add_method) {
            if (!points.empty()) {
                std::vector<Eigen::Vector3d> transformed(points.size());
                std::transform(points.cbegin(), points.cend(), transformed.begin(),
                              [&](const Eigen::Vector3d &point) { return latest_pose.translation() + latest_pose.rotationMatrix() * point; });
                (local_map_.*add_method)(transformed);
            }
        };
        
        add_to_map(downsampled_ground, &VoxelHashMap::AddGroundPoints);
        add_to_map(downsampled_roof, &VoxelHashMap::AddRoofPoints);
        add_to_map(downsampled_wall, &VoxelHashMap::AddWallPoints);
        add_to_map(downsampled_edge, &VoxelHashMap::AddEdgePoints);
        add_to_map(downsampled_non_planar, &VoxelHashMap::AddNonPlanarPoints);
    }
    
    return result;
}

// Overloaded version without timestamps
damm_loam::Vector3dVectorTuple damm_loam::RegisterFrame(const std::vector<Eigen::Vector3d> &ground_points,
                                                    const std::vector<Eigen::Vector3d> &roof_points,
                                                    const std::vector<Eigen::Vector3d> &wall_points,
                                                    const std::vector<Eigen::Vector3d> &edge_points,
                                                    const std::vector<Eigen::Vector3d> &non_planar_points) {
    // Process each point cloud type individually and call version with timestamps (empty timestamps)
    return RegisterFrame(ground_points, roof_points, wall_points, edge_points, non_planar_points, std::vector<double>());
}

double damm_loam::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d damm_loam::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool damm_loam::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

}  // namespace damm_loam::pipeline