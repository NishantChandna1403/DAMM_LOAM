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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>
#include <iostream>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};
struct DegeneracyAnalysis {
    Eigen::Matrix3d translation_eigenvectors;
    Eigen::Matrix3d rotation_eigenvectors;
    Eigen::Vector3d translation_eigenvalues;
    Eigen::Vector3d rotation_eigenvalues;
    std::vector<bool> degenerate_translation_directions;
    std::vector<bool> degenerate_rotation_directions;
    bool has_degeneracy;
    
    DegeneracyAnalysis() : has_degeneracy(false) {
        translation_eigenvectors.setZero();
        rotation_eigenvectors.setZero();
        translation_eigenvalues.setZero();
        rotation_eigenvalues.setZero();
        degenerate_translation_directions.resize(3, false);
        degenerate_rotation_directions.resize(3, false);
    }
};
struct LocalizabilityAnalysis {
    Eigen::MatrixXd Ft;  // N x 3 matrix of normals for planar points
    Eigen::MatrixXd It;  // N x 3 matrix of localizability contributions
    Eigen::MatrixXd It_weighted;  // N x 3 matrix of eigenvalue-weighted localizability contributions
    Eigen::VectorXd weight_matrix;  // N x 1 weight matrix for each planar point
    
    LocalizabilityAnalysis() {}
    

void NormalizeEigenvalues(Eigen::Vector3d& eigenvalues) {
    double max_val = eigenvalues.maxCoeff();
    
    // Avoid division by zero
    if (max_val > 1e-12) {
        eigenvalues = eigenvalues / max_val;
    } else {
        eigenvalues.setOnes();
    }
}
void ComputeLocalizabilityContributions(const std::vector<Eigen::Vector3d>& normals,
                                        const Eigen::Matrix3d& translation_eigenvectors,
                                        const Eigen::Vector3d& translation_eigenvalues) {
    // Create Ft matrix (N x 3) from normals
    size_t N = normals.size();
    Ft.resize(static_cast<int>(N), 3);
    
    for (size_t i = 0; i < N; ++i) {
        Ft.row(static_cast<int>(i)) = normals[i].transpose();
    }
    
    // Compute It = Ft * V_t (localizability contributions)
    It = (Ft * translation_eigenvectors).cwiseAbs();
    
    // Normalize eigenvalues for weighting
    Eigen::Vector3d normalized_eigenvalues = translation_eigenvalues;
    NormalizeEigenvalues(normalized_eigenvalues);
    
    // Multiply each column of It by the corresponding normalized eigenvalue
    It_weighted = It;
    for (int col = 0; col < 3; ++col) {
        It_weighted.col(col) *= normalized_eigenvalues(col);
    }
    
    // Compute weight matrix with threshold
    weight_matrix.resize(static_cast<int>(N));
    for (int i = 0; i < static_cast<int>(N); ++i) {
        double sum_of_squares = 0.0;
        for (int j = 0; j < 3; ++j) {
            sum_of_squares += square(It_weighted(i, j));
        }
        double computed_weight = std::sqrt(sum_of_squares);
        weight_matrix(i) = (computed_weight < 0.1) ? 0.1 : computed_weight;
    }
}
};
void NormalizeEigenvalues(Eigen::Vector3d& eigenvalues) {
    double max_val = eigenvalues.maxCoeff();
    
    // Avoid division by zero
    if (max_val > 1e-12) {
        eigenvalues = eigenvalues / max_val;
    } else {
        eigenvalues.setOnes();
    }
}
void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}
void EigenAnalysis(const Eigen::Matrix6d& hessian, 
                  Eigen::Matrix3d& translation_eigenvectors33, 
                  Eigen::Matrix3d& rotation_eigenvectors33,
                  Eigen::Vector3d& translation_eigenvalues,
                  Eigen::Vector3d& rotation_eigenvalues) {
    // Translation block (bottom-right 3x3)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_translation(hessian.bottomRightCorner(3, 3), 
                                                    Eigen::ComputeFullU | Eigen::ComputeFullV);
    translation_eigenvectors33 = svd_translation.matrixU();
    translation_eigenvalues = svd_translation.singularValues();
    
    // Rotation block (top-left 3x3)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_rotation(hessian.topLeftCorner(3, 3), 
                                                    Eigen::ComputeFullU | Eigen::ComputeFullV);
    rotation_eigenvectors33 = svd_rotation.matrixU();
    rotation_eigenvalues = svd_rotation.singularValues();
}
DegeneracyAnalysis DetectDegeneracy(const Eigen::Matrix6d& hessian, 
                                    double eigenvalue_threshold = 1e-6) {
    DegeneracyAnalysis analysis;
    
    // Perform eigenanalysis
    EigenAnalysis(hessian, 
                  analysis.translation_eigenvectors, 
                  analysis.rotation_eigenvectors,
                  analysis.translation_eigenvalues,
                  analysis.rotation_eigenvalues);
    
    // Check for degenerate directions based on eigenvalues
    analysis.has_degeneracy = false;
    
    // Check translation directions
    for (int i = 0; i < 3; ++i) {
        if (analysis.translation_eigenvalues(i) < eigenvalue_threshold) {
            analysis.degenerate_translation_directions[i] = true;
            analysis.has_degeneracy = true;
        }
    }
    
    // Check rotation directions
    for (int i = 0; i < 3; ++i) {
        if (analysis.rotation_eigenvalues(i) < eigenvalue_threshold) {
            analysis.degenerate_rotation_directions[i] = true;
            analysis.has_degeneracy = true;
        }
    }
    
    return analysis;
}


//Build the linear system for the GenZ-ICP
//Build the linear system for the GenZ-ICP with optional localizability weights
std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    const std::vector<Eigen::Vector3d> &src_planar,
    const std::vector<Eigen::Vector3d> &tgt_planar,
    const std::vector<Eigen::Vector3d> &normals,
    const std::vector<Eigen::Vector3d> &src_non_planar,
    const std::vector<Eigen::Vector3d> &tgt_non_planar,
    double kernel,
    double alpha,
    const Eigen::VectorXd& planar_weights = Eigen::VectorXd()) {

    struct ResultTuple {
        Eigen::Matrix6d JTJ;
        Eigen::Vector6d JTr;

        ResultTuple() : JTJ(Eigen::Matrix6d::Zero()), JTr(Eigen::Vector6d::Zero()) {}

        ResultTuple operator+(const ResultTuple &other) const {
            ResultTuple result;
            result.JTJ = JTJ + other.JTJ;
            result.JTr = JTr + other.JTr;
            return result;
        }
    };

    // Point-to-Plane Jacobian and Residual
    auto compute_jacobian_and_residual_planar = [&](auto i) {
        double r_planar = (src_planar[i] - tgt_planar[i]).dot(normals[i]); // residual
        Eigen::Matrix<double, 1, 6> J_planar; // Jacobian matrix
        J_planar.block<1, 3>(0, 0) = normals[i].transpose(); 
        J_planar.block<1, 3>(0, 3) = (src_planar[i].cross(normals[i])).transpose();
        return std::make_tuple(J_planar, r_planar);
    };

    // Point-to-Point Jacobian and Residual
    auto compute_jacobian_and_residual_non_planar = [&](auto i) {
        const Eigen::Vector3d r_non_planar = src_non_planar[i] - tgt_non_planar[i];
        Eigen::Matrix3_6d J_non_planar;
        J_non_planar.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_non_planar.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(src_non_planar[i]);
        return std::make_tuple(J_non_planar, r_non_planar);
    };

    double kernel_squared = kernel * kernel;
    auto compute = [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
        auto Weight = [&](double residual_squared) {
            return kernel_squared / square(kernel + residual_squared);
        };
        auto &[JTJ_private, JTr_private] = J;
        for (size_t i = r.begin(); i < r.end(); ++i) {
            if (i < src_planar.size()) { // Point-to-Plane
                const auto &[J_planar, r_planar] = compute_jacobian_and_residual_planar(i);
                double w_planar = Weight(r_planar * r_planar);
                
                // Apply localizability weight if available
                if (planar_weights.size() > 0 && static_cast<int>(i) < planar_weights.size()) {
                    w_planar *= planar_weights(static_cast<int>(i));
                }
                
                JTJ_private.noalias() += alpha * J_planar.transpose() * w_planar * J_planar;
                JTr_private.noalias() += alpha * J_planar.transpose() * w_planar * r_planar;
            } else { // Point-to-Point
                size_t index = i - src_planar.size();
                if (index < src_non_planar.size()) {
                    const auto &[J_non_planar, r_non_planar] = compute_jacobian_and_residual_non_planar(index);
                    const double w_non_planar = Weight(r_non_planar.squaredNorm());
                    JTJ_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * J_non_planar;
                    JTr_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * r_non_planar;
                }
            }
        }
        return J;
    };

    size_t total_size = src_planar.size() + src_non_planar.size();
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, total_size),
        ResultTuple(),
        compute,
        [](const ResultTuple &a, const ResultTuple &b) {
            return a + b;
        });

    return std::make_tuple(JTJ, JTr);
}

} 

void VisualizeStatus(size_t planar_count, size_t non_planar_count, double alpha,const DegeneracyAnalysis& analysis) {
    const int bar_width = 52;
    const std::string planar_color = "\033[1;38;2;0;119;187m";
    const std::string non_planar_color = "\033[1;38;2;238;51;119m";
    const std::string alpha_color = "\033[1;32m";
    const std::string degeneracy_color = "\033[1;31m";
    const std::string good_color = "\033[1;32m";

    printf("\033[2J\033[1;1H"); // Clear terminal
    std::cout << "====================== GenZ-ICP ======================\n";
    std::cout << non_planar_color << "# of non-planar points: " << non_planar_count << ", ";
    std::cout << planar_color << "# of planar points: " << planar_count << "\033[0m\n";

    std::cout << "Unstructured  <-----  ";
    std::cout << alpha_color << "alpha: " << std::fixed << std::setprecision(3) << alpha << "\033[0m";
    std::cout << "  ----->  Structured\n";

    const int alpha_location = static_cast<int>(bar_width * alpha); 
    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i == alpha_location) {
            std::cout << "\033[1;32m█\033[0m"; 
        } else {
            std::cout << "-"; 
        }
    }
    std::cout << "\n============== Degeneracy Analysis ==============\n";
    if (analysis.has_degeneracy) {
        std::cout << degeneracy_color << "⚠️  DEGENERACY DETECTED ⚠️\033[0m\n";
    } else {
        std::cout << good_color << "✅ No degeneracy detected\033[0m\n";
    }
    
    // Normalize eigenvalues for display
    Eigen::Vector3d normalized_translation = analysis.translation_eigenvalues;
    Eigen::Vector3d normalized_rotation = analysis.rotation_eigenvalues;
    NormalizeEigenvalues(normalized_translation);
    NormalizeEigenvalues(normalized_rotation);
    
    // Translation degeneracy
    std::cout << "Translation: ";
    const std::vector<std::string> trans_labels = {"X", "Y", "Z"};
    for (int i = 0; i < 3; ++i) {
        if (analysis.degenerate_translation_directions[i]) {
            std::cout << degeneracy_color << trans_labels[i] << "(!) ";
        } else {
            std::cout << good_color << trans_labels[i] << " ";
        }
    }
    std::cout << "\033[0m| Eigenvalues: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << std::scientific << std::setprecision(2) << analysis.translation_eigenvalues(i) << " ";
    }
    std::cout << "\n";
    std::cout << "             | Normalized: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(3) << normalized_translation(i) << " ";
    }
    std::cout << "\n";

    double lambda_min = analysis.translation_eigenvalues.minCoeff();
    double lambda_max = analysis.translation_eigenvalues.maxCoeff();
    std::cout << "condition number: " << std::scientific << std::setprecision(2) << (lambda_max / lambda_min) << "\n";

    std::cout << "]\n";
    std::cout.flush();
}
  // namespace

namespace genz_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion)
    : max_num_iterations_(max_num_iteration), 
      convergence_criterion_(convergence_criterion) {}


std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> Registration::RegisterFrame(
    const std::vector<Eigen::Vector3d> &ground_points,
    const std::vector<Eigen::Vector3d> &roof_points,
    const std::vector<Eigen::Vector3d> &wall_points,
    const std::vector<Eigen::Vector3d> &edge_points,
    const std::vector<Eigen::Vector3d> &non_planar_points,
    const VoxelHashMap &voxel_map,
    const Sophus::SE3d &initial_guess,
    double max_correspondence_distance,
    double kernel) {
    
    // Combine planar points (ground, roof, wall)
    std::vector<Eigen::Vector3d> current_planar;
    current_planar.reserve(ground_points.size() + roof_points.size() + wall_points.size());
    current_planar.insert(current_planar.end(), ground_points.begin(), ground_points.end());
    current_planar.insert(current_planar.end(), roof_points.begin(), roof_points.end());
    current_planar.insert(current_planar.end(), wall_points.begin(), wall_points.end());
    
    // Combine non-planar points (edge + non_planar)
    std::vector<Eigen::Vector3d> current_non_planar;
    current_non_planar.reserve(edge_points.size() + non_planar_points.size());
    current_non_planar.insert(current_non_planar.end(), edge_points.begin(), edge_points.end());
    current_non_planar.insert(current_non_planar.end(), non_planar_points.begin(), non_planar_points.end());
    
    if (voxel_map.Empty()) return std::make_tuple(initial_guess, current_planar, current_non_planar);
    // Apply initial guess transformation
    TransformPoints(initial_guess, current_planar);
    TransformPoints(initial_guess, current_non_planar);
    // GenZ-ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    std::vector<Eigen::Vector3d> final_planar_points;
    std::vector<Eigen::Vector3d> final_non_planar_points;
    DegeneracyAnalysis degeneracy_analysis;
    LocalizabilityAnalysis localizability_analysis;
    
    // Create mutable copies to work with
    std::vector<Eigen::Vector3d> mutable_ground = ground_points;
    std::vector<Eigen::Vector3d> mutable_roof = roof_points;
    std::vector<Eigen::Vector3d> mutable_wall = wall_points;
    std::vector<Eigen::Vector3d> mutable_edge = edge_points;
    std::vector<Eigen::Vector3d> mutable_non_planar = non_planar_points;
    
    // Apply initial transformation to all point types
    TransformPoints(initial_guess, mutable_ground);
    TransformPoints(initial_guess, mutable_roof);
    TransformPoints(initial_guess, mutable_wall);
    TransformPoints(initial_guess, mutable_edge);
    TransformPoints(initial_guess, mutable_non_planar);
    
    for (int j = 0; j < max_num_iterations_; ++j) {
        auto [src_planar, tgt_planar, normals, src_non_planar, tgt_non_planar, planar_count, non_planar_count] =
            voxel_map.GetCorrespondences(mutable_ground, mutable_roof, mutable_wall, mutable_edge, mutable_non_planar, max_correspondence_distance);
        double alpha = static_cast<double>(planar_count) / static_cast<double>(planar_count + non_planar_count);
        const auto &[JTJ_temp, JTr_temp] = BuildLinearSystem(src_planar, tgt_planar, normals, src_non_planar, tgt_non_planar, kernel, alpha);
        degeneracy_analysis = DetectDegeneracy(JTJ_temp);
        if (!normals.empty()) {
            localizability_analysis.ComputeLocalizabilityContributions(normals, 
                                                                     degeneracy_analysis.translation_eigenvectors,
                                                                     degeneracy_analysis.translation_eigenvalues);
        }
        const auto &[JTJ, JTr] = BuildLinearSystem(src_planar, tgt_planar, normals, src_non_planar, tgt_non_planar, kernel, alpha, localizability_analysis.weight_matrix);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);   

        
        // Transform all individual point types
        TransformPoints(estimation, mutable_ground);
        TransformPoints(estimation, mutable_roof);
        TransformPoints(estimation, mutable_wall);
        TransformPoints(estimation, mutable_edge);
        TransformPoints(estimation, mutable_non_planar);
        
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < convergence_criterion_ || j == max_num_iterations_ - 1) {
            VisualizeStatus(planar_count, non_planar_count, alpha,degeneracy_analysis);
            final_planar_points = src_planar;
            final_non_planar_points = src_non_planar;
            break;
        }
    }

    // // Spit the final transformation
    return std::make_tuple(T_icp * initial_guess, final_planar_points, final_non_planar_points);
}


}  // namespace genz_icp