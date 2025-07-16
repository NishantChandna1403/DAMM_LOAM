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
#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

// This parameters are not intended to be changed, therefore we do not expose it
namespace {
struct ResultTuple { 
    ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};

static const std::array<Eigen::Vector3i, 27> voxel_shifts{
    Eigen::Vector3i(0, 0, 0),   Eigen::Vector3i(1, 0, 0),   Eigen::Vector3i(-1, 0, 0),
    Eigen::Vector3i(0, 1, 0),   Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 0, 1),
    Eigen::Vector3i(0, 0, -1),  Eigen::Vector3i(1, 1, 0),   Eigen::Vector3i(1, -1, 0),
    Eigen::Vector3i(-1, 1, 0),  Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 0, 1),
    Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, 1),  Eigen::Vector3i(-1, 0, -1),
    Eigen::Vector3i(0, 1, 1),   Eigen::Vector3i(0, 1, -1),  Eigen::Vector3i(0, -1, 1),
    Eigen::Vector3i(0, -1, -1), Eigen::Vector3i(1, 1, 1),   Eigen::Vector3i(1, 1, -1),
    Eigen::Vector3i(1, -1, 1),  Eigen::Vector3i(1, -1, -1), Eigen::Vector3i(-1, 1, 1),
    Eigen::Vector3i(-1, 1, -1), Eigen::Vector3i(-1, -1, 1), Eigen::Vector3i(-1, -1, -1)
};

static const size_t min_neighbors_for_normal_estimation = 5; 
}  // namespace

namespace genz_icp {

std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> VoxelHashMap::GetClosestNeighbor(
    const Eigen::Vector3d &query) const {
    Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

    double closest_squared_distance = std::numeric_limits<double>::max();
    size_t n_neighbors = 0;
    
    auto kx = static_cast<int>(query.x() / voxel_size_);
    auto ky = static_cast<int>(query.y() / voxel_size_);
    auto kz = static_cast<int>(query.z() / voxel_size_);

    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto &voxel_shift) {
        Voxel voxel(kx+voxel_shift.x(), ky+voxel_shift.y(), kz+voxel_shift.z());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            const auto &points = search->second.points;
            std::for_each(points.cbegin(), points.cend(), [&](const auto &neighbor){
                double squared_distance = (neighbor - query).squaredNorm();
                if (squared_distance < closest_squared_distance){
                    closest_neighbor = neighbor;
                    closest_squared_distance = squared_distance;
                }
                centroid += neighbor;
                covariance += neighbor*neighbor.transpose();
                n_neighbors++;
            });
        }
    });

    if (n_neighbors >= min_neighbors_for_normal_estimation){
        centroid /= static_cast<double>(n_neighbors);
        covariance /= static_cast<double>(n_neighbors);
        covariance -= centroid*centroid.transpose();
    }
    double closest_distance = (n_neighbors > 0) ? std::sqrt(closest_squared_distance) : std::numeric_limits<double>::max();
    return std::make_tuple(closest_neighbor, n_neighbors, covariance, closest_distance);
}

std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> VoxelHashMap::GetClosestNeighborFromMap(
    const Eigen::Vector3d &query, const tsl::robin_map<Voxel, VoxelBlock, VoxelHash> &target_map) const {
    Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

    double closest_squared_distance = std::numeric_limits<double>::max();
    size_t n_neighbors = 0;
    
    auto kx = static_cast<int>(query.x() / voxel_size_);
    auto ky = static_cast<int>(query.y() / voxel_size_);
    auto kz = static_cast<int>(query.z() / voxel_size_);

    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto &voxel_shift) {
        Voxel voxel(kx+voxel_shift.x(), ky+voxel_shift.y(), kz+voxel_shift.z());
        auto search = target_map.find(voxel);
        if (search != target_map.end()) {
            const auto &points = search->second.points;
            std::for_each(points.cbegin(), points.cend(), [&](const auto &neighbor){
                double squared_distance = (neighbor - query).squaredNorm();
                if (squared_distance < closest_squared_distance){
                    closest_neighbor = neighbor;
                    closest_squared_distance = squared_distance;
                }
                centroid += neighbor;
                covariance += neighbor*neighbor.transpose();
                n_neighbors++;
            });
        }
    });

    if (n_neighbors >= min_neighbors_for_normal_estimation){
        centroid /= static_cast<double>(n_neighbors);
        covariance /= static_cast<double>(n_neighbors);
        covariance -= centroid*centroid.transpose();
    }
    double closest_distance = (n_neighbors > 0) ? std::sqrt(closest_squared_distance) : std::numeric_limits<double>::max();
    return std::make_tuple(closest_neighbor, n_neighbors, covariance, closest_distance);
}

std::pair<bool, Eigen::Vector3d> VoxelHashMap::DeterminePlanarity(
    const Eigen::Matrix3d &covariance) const {
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    // Compute the normal as the eigenvector of the smallest eigenvalue
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance, Eigen::ComputeEigenvectors);
    normal = solver.eigenvectors().col(0);
    normal = normal.normalized();

    // Planarity check
    const auto &eigenvalues = solver.eigenvalues();
    double lambda3 = eigenvalues[0];
    double lambda2 = eigenvalues[1];
    double lambda1 = eigenvalues[2];

    // Check if the surface is planar
    bool is_planar = (lambda3 / (lambda1 + lambda2 + lambda3)) < planarity_threshold_;

    return {is_planar, normal};
}

// New GetCorrespondences method for 5 separate point cloud types
VoxelHashMap::Vector3dVectorTuple7 VoxelHashMap::GetCorrespondences(
    const Vector3dVector &ground_points,
    const Vector3dVector &roof_points,
    const Vector3dVector &wall_points,
    const Vector3dVector &edge_points,
    const Vector3dVector &non_planar_points,
    double max_correspondance_distance) const {

    struct ResultTuple {
        Vector3dVector source;
        Vector3dVector target;
        Vector3dVector normals;
        Vector3dVector non_planar_source;
        Vector3dVector non_planar_target;
        size_t planar_count = 0;
        size_t non_planar_count = 0;

        ResultTuple() = default;
        ResultTuple(size_t n) {
            source.reserve(n);
            target.reserve(n);
            normals.reserve(n);
            non_planar_source.reserve(n);
            non_planar_target.reserve(n);
        }

        ResultTuple operator+(ResultTuple other) const {
            ResultTuple result(*this);
            result.source.insert(result.source.end(),
                std::make_move_iterator(other.source.begin()), std::make_move_iterator(other.source.end()));
            result.target.insert(result.target.end(),
                std::make_move_iterator(other.target.begin()), std::make_move_iterator(other.target.end()));
            result.normals.insert(result.normals.end(),
                std::make_move_iterator(other.normals.begin()), std::make_move_iterator(other.normals.end()));
            result.non_planar_source.insert(result.non_planar_source.end(),
                std::make_move_iterator(other.non_planar_source.begin()), std::make_move_iterator(other.non_planar_source.end()));
            result.non_planar_target.insert(result.non_planar_target.end(),
                std::make_move_iterator(other.non_planar_target.begin()), std::make_move_iterator(other.non_planar_target.end()));
            result.planar_count += other.planar_count;
            result.non_planar_count += other.non_planar_count;
            return result;
        }
    };

    // Helper function to process planar points (ground, roof, wall) and add to planar correspondences
    auto process_planar_points = [&](const Vector3dVector &points, ResultTuple &result) {
        for (const auto &point : points) {
            // Use the original GetClosestNeighbor which searches in the main map_
            const auto &[closest_neighbor, n_neighbors, covariance, closest_distance] = GetClosestNeighbor(point);
            if (closest_distance > max_correspondance_distance) continue;
            
            if (n_neighbors >= min_neighbors_for_normal_estimation) {
                // Compute normal without planarity check since we know these are planar by classification
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance, Eigen::ComputeEigenvectors);
                Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
                result.source.emplace_back(point);
                result.target.emplace_back(closest_neighbor);
                result.normals.emplace_back(normal);
                result.planar_count++;
            }
        }
    };

    // Helper function to process non-planar points (edge, non_planar) and add to non-planar correspondences
    auto process_non_planar_points = [&](const Vector3dVector &points, ResultTuple &result) {
        for (const auto &point : points) {
            // Use the original GetClosestNeighbor which searches in the main map_
            const auto &[closest_neighbor, n_neighbors, covariance, closest_distance] = GetClosestNeighbor(point);
            if (closest_distance > max_correspondance_distance) continue;
            
            result.non_planar_source.emplace_back(point);
            result.non_planar_target.emplace_back(closest_neighbor);
            result.non_planar_count++;
        }
    };

    // Calculate total size for reserve
    size_t total_size = ground_points.size() + roof_points.size() + wall_points.size() + 
                       edge_points.size() + non_planar_points.size();
    ResultTuple final_result(total_size);

    // Process planar types (ground, roof, wall) - all go into planar correspondences
    process_planar_points(ground_points, final_result);
    process_planar_points(roof_points, final_result);
    process_planar_points(wall_points, final_result);
    
    // Process non-planar types (edge, non_planar) - all go into non-planar correspondences
    process_non_planar_points(edge_points, final_result);
    process_non_planar_points(non_planar_points, final_result);

    return std::make_tuple(final_result.source, final_result.target, final_result.normals, 
                          final_result.non_planar_source, final_result.non_planar_target, 
                          final_result.planar_count, final_result.non_planar_count);
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    
    // Calculate total size estimate from main map
    size_t total_size = max_points_per_voxel_ * map_.size();
    points.reserve(total_size);
    
    // Collect points from main target map
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.emplace_back(point);
        }
    }
    
    return points;
}

void VoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.insert({voxel, VoxelBlock(point, max_points_per_voxel_)});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = map_cleanup_radius_ * map_cleanup_radius_;
    
    // Clean up main target map
    for (auto it = map_.begin(); it != map_.end();) {
        if ((it->second.points.front() - origin).squaredNorm() > (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

void VoxelHashMap::AddGroundPoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

void VoxelHashMap::AddRoofPoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

void VoxelHashMap::AddHorizontalPoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

void VoxelHashMap::AddWallPoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

void VoxelHashMap::AddEdgePoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

void VoxelHashMap::AddNonPlanarPoints(const std::vector<Eigen::Vector3d> &points) {
    AddPoints(points); // Add to main target map for correspondence search
}

}  // namespace genz_icp