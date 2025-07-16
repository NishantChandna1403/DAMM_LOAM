#ifndef UTILS_H
#define UTILS_H

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <fstream>
#include <string.h>
#include <thread>
#include <mutex>
#include <deque>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

//Messages
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

//ROS
#include <ros/ros.h>

//EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>

//OpenCV (needed for normalExtraction.cpp)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct ProjectedPoint {
    PCL_ADD_POINT4D;
    float intensity;
    float range = 0.0f;
    int valid = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(ProjectedPoint,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, range, range) (int, valid, valid)
)

struct PointsWithNormals {
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    float intensity;
    float range = 0.0f;
    int valid = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointsWithNormals,
    (float, x, x) (float, y, y) (float, z, z) (float, normal_x, normal_x) (float, normal_y, normal_y) (float, normal_z, normal_z)
    (float, intensity, intensity) (float, range, range) (int, valid, valid)
)

#endif
