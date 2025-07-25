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

// damm-loam
#include "damm_loam/pipeline/damm_loam.hpp"

// ROS
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// Message filters for synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <string>
#include <memory>

namespace damm_loam_ros {

// Synchronization policy for 5 topics
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> SyncPolicy;

class OdometryServer {
public:
    /// OdometryServer constructor
    OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);

private:
    /// Register new frame with 5 topic inputs
    void PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &ground_msg,
                           const sensor_msgs::PointCloud2::ConstPtr &roof_msg,
                           const sensor_msgs::PointCloud2::ConstPtr &wall_msg,
                           const sensor_msgs::PointCloud2::ConstPtr &edge_msg,
                           const sensor_msgs::PointCloud2::ConstPtr &non_planar_msg);

    /// Register new frame
    void RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &ground_msg,
                       const sensor_msgs::PointCloud2::ConstPtr &roof_msg,
                       const sensor_msgs::PointCloud2::ConstPtr &wall_msg,
                       const sensor_msgs::PointCloud2::ConstPtr &edge_msg,
                       const sensor_msgs::PointCloud2::ConstPtr &non_planar_msg);

    /// Stream the estimated pose to ROS
    void PublishOdometry(const Sophus::SE3d &pose,
                         const ros::Time &stamp,
                         const std::string &cloud_frame_id);

    /// Stream the debugging point clouds for visualization (if required)
    void PublishClouds(const ros::Time &stamp,
                       const std::string &cloud_frame_id,
                       const std::vector<Eigen::Vector3d> &ground_points,
                       const std::vector<Eigen::Vector3d> &roof_points,
                       const std::vector<Eigen::Vector3d> &wall_points,
                       const std::vector<Eigen::Vector3d> &edge_points,
                       const std::vector<Eigen::Vector3d> &non_planar_points);

    /// Utility function to compute transformation using tf tree
    Sophus::SE3d LookupTransform(const std::string &target_frame,
                                 const std::string &source_frame) const;

    /// Ros node stuff
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    int queue_size_{1};

    /// Tools for broadcasting TFs.
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::Buffer tf2_buffer_;
    tf2_ros::TransformListener tf2_listener_;
    bool publish_odom_tf_;
    bool publish_debug_clouds_;

    /// Data subscribers.
    message_filters::Subscriber<sensor_msgs::PointCloud2> ground_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> roof_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> wall_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> edge_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> non_planar_sub_;
    
    /// Synchronization policy for 5 topics
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
        sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> SyncPolicy;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    /// Data publishers.
    ros::Publisher odom_publisher_;
    ros::Publisher map_publisher_;
    ros::Publisher traj_publisher_;
    ros::Publisher ground_points_publisher_;
    ros::Publisher roof_points_publisher_;
    ros::Publisher wall_points_publisher_;
    ros::Publisher edge_points_publisher_;
    ros::Publisher non_planar_points_publisher_;
    nav_msgs::Path path_msg_;

    /// damm-loam
    damm_loam::pipeline::damm_loam odometry_;
    damm_loam::pipeline::DammConfig config_;

    /// Global/map coordinate frame.
    std::string odom_frame_{"odom"};
    std::string base_frame_{};
};

}  // namespace damm_loam_ros
