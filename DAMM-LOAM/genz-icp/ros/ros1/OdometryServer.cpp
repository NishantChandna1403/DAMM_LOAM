#include <Eigen/Core>
#include <memory>
#include <utility>
#include <vector>

// GenZ-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// GenZ-ICP
#include "genz_icp/pipeline/GenZICP.hpp"

// ROS 1 headers
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

// Message filters for synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

namespace genz_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh), tf2_listener_(tf2_ros::TransformListener(tf2_buffer_)) {
    pnh_.param("base_frame", base_frame_, base_frame_);
    pnh_.param("odom_frame", odom_frame_, odom_frame_);
    pnh_.param("publish_odom_tf", publish_odom_tf_, false);
    pnh_.param("visualize", publish_debug_clouds_, publish_debug_clouds_);
    pnh_.param("max_range", config_.max_range, config_.max_range);
    pnh_.param("min_range", config_.min_range, config_.min_range);
    pnh_.param("deskew", config_.deskew, config_.deskew);
    pnh_.param("voxel_size", config_.voxel_size, config_.max_range / 100.0);
    pnh_.param("map_cleanup_radius", config_.map_cleanup_radius, config_.max_range);
    pnh_.param("planarity_threshold", config_.planarity_threshold, config_.planarity_threshold);
    pnh_.param("max_points_per_voxel", config_.max_points_per_voxel, config_.max_points_per_voxel);
    pnh_.param("desired_num_voxelized_points", config_.desired_num_voxelized_points, config_.desired_num_voxelized_points);
    pnh_.param("initial_threshold", config_.initial_threshold, config_.initial_threshold);
    pnh_.param("min_motion_th", config_.min_motion_th, config_.min_motion_th);
    pnh_.param("max_num_iterations", config_.max_num_iterations, config_.max_num_iterations);
    pnh_.param("convergence_criterion", config_.convergence_criterion, config_.convergence_criterion);
    if (config_.max_range < config_.min_range) {
        ROS_WARN("[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
        config_.min_range = 0.0;
    }

    // Construct the main GenZ-ICP odometry node
    // Create GenZConfig struct from our parameters
    genz_icp::pipeline::GenZConfig genz_config;
    genz_config.max_range = config_.max_range;
    genz_config.min_range = config_.min_range;
    genz_config.deskew = config_.deskew;
    genz_config.voxel_size = config_.voxel_size;
    genz_config.map_cleanup_radius = config_.map_cleanup_radius;
    genz_config.planarity_threshold = config_.planarity_threshold;
    genz_config.max_points_per_voxel = config_.max_points_per_voxel;
    genz_config.desired_num_voxelized_points = config_.desired_num_voxelized_points;
    genz_config.initial_threshold = config_.initial_threshold;
    genz_config.min_motion_th = config_.min_motion_th;
    genz_config.max_num_iterations = config_.max_num_iterations;
    genz_config.convergence_criterion = config_.convergence_criterion;
    
    // Initialize odometry with the config
    odometry_ = genz_icp::pipeline::GenZICP(genz_config);

    // Initialize synchronized subscribers for 5 topics
    ground_sub_.subscribe(nh_, "/nv_liom/ground_cloud", queue_size_);
    roof_sub_.subscribe(nh_, "/nv_liom/roof_cloud", queue_size_);
    wall_sub_.subscribe(nh_, "/nv_liom/wall_cloud", queue_size_);
    edge_sub_.subscribe(nh_, "/nv_liom/edge_cloud", queue_size_);
    non_planar_sub_.subscribe(nh_, "/nv_liom/non_planar_cloud", queue_size_);
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(queue_size_), ground_sub_, roof_sub_, wall_sub_, edge_sub_, non_planar_sub_);
    sync_->setMaxIntervalDuration(ros::Duration(0.1)); // Set slop to 0.1s
    sync_->registerCallback(boost::bind(&OdometryServer::PointCloudCallback, this, _1, _2, _3, _4, _5));

    // Initialize publishers
    odom_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/genz/odometry", queue_size_);
    traj_publisher_ = pnh_.advertise<nav_msgs::Path>("/genz/trajectory", queue_size_);
    if (publish_debug_clouds_) {
        map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/local_map", queue_size_);
        ground_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/ground_points", queue_size_);
        roof_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/roof_points", queue_size_);
        wall_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/wall_points", queue_size_);
        edge_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/edge_points", queue_size_);
        non_planar_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/non_planar_points", queue_size_);
    }
    // Initialize the transform buffer
    tf2_buffer_.setUsingDedicatedThread(true);
    path_msg_.header.frame_id = odom_frame_;

    ROS_INFO("GenZ-ICP ROS 1 Odometry Node Initialized");
}

Sophus::SE3d OdometryServer::LookupTransform(const std::string &target_frame,
                                             const std::string &source_frame) const {
    std::string err_msg;
    if (tf2_buffer_._frameExists(source_frame) &&  //
        tf2_buffer_._frameExists(target_frame) &&  //
        tf2_buffer_.canTransform(target_frame, source_frame, ros::Time(0), &err_msg)) {
        try {
            auto tf = tf2_buffer_.lookupTransform(target_frame, source_frame, ros::Time(0));
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            ROS_WARN("%s", ex.what());
        }
    }
    ROS_WARN("Failed to find tf between %s and %s. Reason=%s", target_frame.c_str(),
             source_frame.c_str(), err_msg.c_str());
    return {};
}

void OdometryServer::PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &ground_msg,
                                       const sensor_msgs::PointCloud2::ConstPtr &roof_msg,
                                       const sensor_msgs::PointCloud2::ConstPtr &wall_msg,
                                       const sensor_msgs::PointCloud2::ConstPtr &edge_msg,
                                       const sensor_msgs::PointCloud2::ConstPtr &non_planar_msg) {
    ROS_INFO("Received ground: %u, roof: %u, wall: %u, edge: %u, non-planar: %u points, timestamp: %f",
             ground_msg->width * ground_msg->height, roof_msg->width * roof_msg->height,
             wall_msg->width * wall_msg->height, edge_msg->width * edge_msg->height,
             non_planar_msg->width * non_planar_msg->height,
             ground_msg->header.stamp.toSec());

    try {
        RegisterFrame(ground_msg, roof_msg, wall_msg, edge_msg, non_planar_msg);
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in RegisterFrame: %s", e.what());
    }
}

void OdometryServer::RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &ground_msg,
                                   const sensor_msgs::PointCloud2::ConstPtr &roof_msg,
                                   const sensor_msgs::PointCloud2::ConstPtr &wall_msg,
                                   const sensor_msgs::PointCloud2::ConstPtr &edge_msg,
                                   const sensor_msgs::PointCloud2::ConstPtr &non_planar_msg) {
    const auto cloud_frame_id = ground_msg->header.frame_id;
    const auto ground_points = PointCloud2ToEigen(ground_msg);
    const auto roof_points = PointCloud2ToEigen(roof_msg);
    const auto wall_points = PointCloud2ToEigen(wall_msg);
    const auto edge_points = PointCloud2ToEigen(edge_msg);
    const auto non_planar_points = PointCloud2ToEigen(non_planar_msg);

    ROS_INFO("Converted to %zu ground, %zu roof, %zu wall, %zu edge, %zu non-planar points", 
             ground_points.size(), roof_points.size(), wall_points.size(), 
             edge_points.size(), non_planar_points.size());

    // Skip if all point clouds are empty
    if (ground_points.empty() && roof_points.empty() && wall_points.empty() && 
        edge_points.empty() && non_planar_points.empty()) {
        ROS_WARN("All point clouds are empty, skipping registration");
        return;
    }

    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return GetTimestamps(ground_msg);
    }();
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);

    // Register frame, main entry point to GenZ-ICP pipeline
    try {
        const auto &[registered_planar, registered_non_planar] = odometry_.RegisterFrame(
            ground_points, roof_points, wall_points, edge_points, non_planar_points, timestamps);
        ROS_INFO("GenZ-ICP registration complete");

        // Compute the pose using GenZ, ego-centric to the LiDAR
        const Sophus::SE3d genz_pose = odometry_.poses().back();

        // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
        const auto pose = [&]() -> Sophus::SE3d {
            if (egocentric_estimation) return genz_pose;
            const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id);
            return cloud2base * genz_pose * cloud2base.inverse();
        }();

        // Publish the current estimated pose to ROS msgs
        PublishOdometry(pose, ground_msg->header.stamp, cloud_frame_id);

        // Publishing these clouds is costly, so do it only if debugging
        if (publish_debug_clouds_) {
            PublishClouds(ground_msg->header.stamp, cloud_frame_id, ground_points, roof_points, wall_points, edge_points, non_planar_points);
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in GenZ-ICP pipeline: %s", e.what());
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &pose,
                                     const ros::Time &stamp,
                                     const std::string &cloud_frame_id) {
    // Broadcast the tf
    if (publish_odom_tf_) {
        geometry_msgs::TransformStamped transform_msg;
        transform_msg.header.stamp = stamp;
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
        transform_msg.transform = tf2::sophusToTransform(pose);
        tf_broadcaster_.sendTransform(transform_msg);
    }

    // Publish trajectory msg
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = odom_frame_;
    pose_msg.pose = tf2::sophusToPose(pose);
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_.publish(path_msg_);

    // Publish odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_publisher_.publish(odom_msg);
}

void OdometryServer::PublishClouds(const ros::Time &stamp,
                                   const std::string &cloud_frame_id,
                                   const std::vector<Eigen::Vector3d> &ground_points,
                                   const std::vector<Eigen::Vector3d> &roof_points,
                                   const std::vector<Eigen::Vector3d> &wall_points,
                                   const std::vector<Eigen::Vector3d> &edge_points,
                                   const std::vector<Eigen::Vector3d> &non_planar_points) {
    std_msgs::Header odom_header;
    odom_header.stamp = stamp;
    odom_header.frame_id = odom_frame_;

    // Publish map
    const auto genz_map = odometry_.LocalMap();

    // Always publish point clouds in odometry frame for consistency
    const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);
    
    // Transform point clouds to odometry frame and publish
    auto transform_and_publish = [&](const std::vector<Eigen::Vector3d> &points, 
                                     ros::Publisher &publisher) {
        if (!points.empty()) {
            std::vector<Eigen::Vector3d> transformed_points(points.size());
            std::transform(points.begin(), points.end(), transformed_points.begin(),
                          [&](const Eigen::Vector3d &point) { return cloud2odom * point; });
            publisher.publish(*EigenToPointCloud2(transformed_points, odom_header));
        } else {
            publisher.publish(*EigenToPointCloud2(points, odom_header));
        }
    };
    
    transform_and_publish(ground_points, ground_points_publisher_);
    transform_and_publish(roof_points, roof_points_publisher_);
    transform_and_publish(wall_points, wall_points_publisher_);
    transform_and_publish(edge_points, edge_points_publisher_);
    transform_and_publish(non_planar_points, non_planar_points_publisher_);

    // Publish map in odometry frame
    map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
}

}  // namespace genz_icp_ros

int main(int argc, char **argv) {
    ros::init(argc, argv, "genz_icp");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    genz_icp_ros::OdometryServer node(nh, nh_private);

    ros::spin();

    return 0;
}