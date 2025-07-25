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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace damm_loam {
struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using Vector3dVectorTuple7 = std::tuple<Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, size_t, size_t>;
    using Voxel = Eigen::Vector3i;
    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector3d> points;
        int num_points_;
        VoxelBlock(const Eigen::Vector3d &point, int max_points) : points(), num_points_(max_points) {
            points.reserve(static_cast<size_t>(num_points_));
            points.emplace_back(point);
        }
        inline void AddPoint(const Eigen::Vector3d &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.emplace_back(point);
        }
    };
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, double map_cleanup_radius,
                          double planarity_threshold, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          map_cleanup_radius_(map_cleanup_radius),
          planarity_threshold_(planarity_threshold),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector3dVectorTuple7 GetCorrespondences(const Vector3dVector &planar_points,
                                           const Vector3dVector &non_planar_points,
                                           double max_correspondance_distance) const;
    // New method for 5 separate point cloud types                                           
    Vector3dVectorTuple7 GetCorrespondences(const Vector3dVector &ground_points,
                                           const Vector3dVector &roof_points,
                                           const Vector3dVector &wall_points,
                                           const Vector3dVector &edge_points,
                                           const Vector3dVector &non_planar_points,
                                           double max_correspondance_distance) const;
    inline void Clear() {
        map_.clear();
    }
    inline bool Empty() const { 
        return map_.empty(); 
    }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void AddGroundPoints(const std::vector<Eigen::Vector3d> &points);
    void AddRoofPoints(const std::vector<Eigen::Vector3d> &points);
    void AddHorizontalPoints(const std::vector<Eigen::Vector3d> &points);  // Combined ground+roof
    void AddWallPoints(const std::vector<Eigen::Vector3d> &points);
    void AddEdgePoints(const std::vector<Eigen::Vector3d> &points);
    void AddNonPlanarPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> GetClosestNeighbor(
        const Eigen::Vector3d &query) const;
    std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> GetClosestNeighborFromMap(
        const Eigen::Vector3d &query, const tsl::robin_map<Voxel, VoxelBlock, VoxelHash> &target_map) const;
    std::pair<bool, Eigen::Vector3d> DeterminePlanarity(const Eigen::Matrix3d &covariance) const;


    double voxel_size_;
    double max_distance_;
    double map_cleanup_radius_;
    double planarity_threshold_;
    int max_points_per_voxel_;
    
    // Main hashmap for target correspondence search
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};

}  // namespace damm_loam
