#include "utils.h"

int hor_pixel_num;
int ver_pixel_num;
float hor_fov;
float ver_max;
float ver_min;

float min_dist;
float max_dist;
float hor_resolution;
float ver_resolution;
int show_img;
ros::Publisher pubNormalPointCloud;
ros::Publisher pubClassifiedPointCloud;
ros::Publisher pubGroundPointCloud;
ros::Publisher pubRoofPointCloud;
ros::Publisher pubWallPointCloud;
ros::Publisher pubEdgePointCloud;
ros::Publisher pubNonPlanarPointCloud;

int normal_neighbor;
// ros::Publisher pubNormalPointCloudRGB;

std::string map_save_dir;

std::vector<Eigen::Matrix3f> sp2cart_map;

cv::Mat normal_img;
cv::Mat normal_img_sp;
cv::Mat classification_img;

// Classification types
enum PointClass {
    UNKNOWN = 0,
    GROUND = 1,
    ROOF = 2,
    WALL = 3,      // Any wall surface (X or Y axis dominant)
    EDGE = 4
};

// Classification parameters
float dominance_threshold = 0.85f;  // Threshold for axis dominance
float angle_variance_threshold = 15.0f * M_PI / 180.0f;  // 15 degrees in radians

// Point structure for classified points
struct ClassifiedPoint {
    float x, y, z;
    float intensity;
    float range;
    float normal_x, normal_y, normal_z;
    uint8_t classification;  // 0=UNKNOWN, 1=GROUND, 2=ROOF, 3=WALL, 4=EDGE
    uint8_t valid;
    uint32_t rgb;  // Packed RGB color (standard PCL format)
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(ClassifiedPoint,
    (float, x, x)
    (float, y, y)  
    (float, z, z)
    (float, intensity, intensity)
    (float, range, range)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (uint8_t, classification, classification)
    (uint8_t, valid, valid)
    (uint32_t, rgb, rgb)
)

void index2uv(int index, int & u, int & v) {
    v = index / hor_pixel_num;
    u = index % hor_pixel_num;
    return;
}

int uv2index(int u, int v) {
    return v*hor_pixel_num + u;
}

// Classify pixel based on normal vector analysis of 3x3 neighborhood
PointClass classifyPixel(int u, int v, const pcl::PointCloud<PointsWithNormals>::Ptr & normal_cloud) {
    // Skip edge pixels that don't have full 3x3 neighborhood
    if(u <= 0 || u >= hor_pixel_num-1 || v <= 0 || v >= ver_pixel_num-1) {
        return UNKNOWN;
    }
    
    std::vector<Eigen::Vector3f> valid_normals;
    
    // Collect valid normals from 3x3 neighborhood
    for(int j = -1; j <= 1; ++j) {
        for(int k = -1; k <= 1; ++k) {
            int query_i = uv2index(u+j, v+k);
            if(query_i < normal_cloud->size() && normal_cloud->points[query_i].valid == 1) {
                Eigen::Vector3f normal(
                    normal_cloud->points[query_i].normal_x,
                    normal_cloud->points[query_i].normal_y,
                    normal_cloud->points[query_i].normal_z
                );
                valid_normals.push_back(normal);
            }
        }
    }
    
    // Need at least 3 valid normals for classification
    if(valid_normals.size() < 3) {
        return UNKNOWN;
    }
    
    // Check for edge based on angle variance
    float angle_sum = 0.0f;
    int angle_count = 0;
    for(size_t i = 0; i < valid_normals.size(); ++i) {
        for(size_t j = i+1; j < valid_normals.size(); ++j) {
            float dot_product = valid_normals[i].dot(valid_normals[j]);
            // Clamp dot product to avoid numerical issues with acos
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
            float angle = acos(abs(dot_product));
            angle_sum += angle;
            angle_count++;
        }
    }
    
    float avg_angle = angle_sum / angle_count;
    if(avg_angle > angle_variance_threshold) {
        return EDGE;
    }
    
    // Find the dominant axis for each normal using maximum absolute component
    int wall_votes = 0, z_pos_votes = 0, z_neg_votes = 0;
    
    for(const auto& normal : valid_normals) {
        // Find the axis with maximum absolute component
        float abs_x = abs(normal.x());
        float abs_y = abs(normal.y());
        float abs_z = abs(normal.z());
        
        if(abs_z >= abs_x && abs_z >= abs_y) {
            // Z-axis is dominant
            if(normal.z() > 0) {
                z_pos_votes++;  // Pointing up
            } else {
                z_neg_votes++;  // Pointing down
            }
        }
        else {
            // X or Y axis is dominant - both are wall surfaces
            wall_votes++;
        }
    }
    
    int total_normals = valid_normals.size();
    
    // Classification based on majority voting using dominant axes
    if(z_pos_votes >= (2 * total_normals) / 3) {
        return GROUND;  // Majority normals pointing up (Z+)
    }
    else if(z_neg_votes >= (2 * total_normals) / 3) {
        return ROOF;    // Majority normals pointing down (Z-)
    }
    else if(wall_votes >= (2 * total_normals) / 3) {
        return WALL;    // Majority normals X or Y axis dominant (any wall surface)
    }
    else if((wall_votes + z_pos_votes + z_neg_votes) < total_normals / 3) {
        // Not enough consistent dominance - likely an edge
        return EDGE;
    }
    
    return UNKNOWN;
}

// Get color for classification visualization (BGR for OpenCV images)
cv::Vec3b getClassificationColor(PointClass cls) {
    switch(cls) {
        case GROUND:     return cv::Vec3b(100, 160, 200);  // Slightly lighter brown (BGR)
        case ROOF:       return cv::Vec3b(255, 255, 255);  // Hard white (BGR) - maximum distinction
        case WALL:       return cv::Vec3b(180, 200, 200);  // Darker cream (BGR)
        case EDGE:       return cv::Vec3b(19, 42, 87);     // Brown (BGR) - reverted to earlier
        default:         return cv::Vec3b(0, 0, 0);        // Black (BGR)
    }
}

// Get RGB color for point cloud (packed RGB format)
uint32_t getClassificationColorRGB(PointClass cls) {
    uint8_t r, g, b;
    switch(cls) {
        case GROUND:     r = 200; g = 160; b = 100; break; // Slightly lighter brown (RGB)
        case ROOF:       r = 200; g = 200; b = 200; break;    // Dark blue (RGB) - more distinct
        case WALL:       r = 200; g = 200; b = 180; break; // Darker cream (RGB)
        case EDGE:       r = 87; g = 42; b = 19; break;    // Brown (RGB) - reverted to earlier
        default:         r = 0; g = 0; b = 0; break;       // Black (RGB)
    }
    // Pack RGB into 32-bit value (standard PCL format)
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

void OnInitialization() {
    sp2cart_map = std::vector<Eigen::Matrix3f>(hor_pixel_num * ver_pixel_num, Eigen::Matrix3f::Zero(3,3));

    for(int i = 0; i < hor_pixel_num * ver_pixel_num; ++i) {
        int u, v;
        index2uv(i, u, v);
        float psi = M_PI - float(u) * hor_resolution;
        // float theta = (M_PI - (ver_fov * M_PI/180.0))/2.0 + float(v) * ver_resolution;

        float theta = M_PI/2.0 - (ver_max * M_PI/180.0) + (float(v) * ver_resolution);

        Eigen::Matrix3f & mat = sp2cart_map[i];

        mat(0, 0) = -sin(psi);
        mat(0, 1) = cos(psi)*cos(theta);
        mat(0, 2) = cos(psi)*sin(theta);

        mat(1, 0) = cos(psi);
        mat(1, 1) = sin(psi)*cos(theta);
        mat(1, 2) = sin(psi)*sin(theta);
        
        mat(2, 0) = 0.0;
        mat(2, 1) = -sin(theta);
        mat(2, 2) = cos(theta);
    }
}

void NormalExtraction(const pcl::PointCloud<ProjectedPoint>::Ptr & projected_cloud, 
                      pcl::PointCloud<PointsWithNormals>::Ptr &normal_cloud,
                      pcl::PointCloud<ClassifiedPoint>::Ptr &classified_cloud) {

    if(show_img == 1) {
        normal_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(0, 0, 0));
        normal_img_sp = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(0, 0, 0));
        classification_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    #pragma omp parallel for num_threads(4)
    for(int i = 0; i < projected_cloud->size(); ++i) {
        int u, v;
        index2uv(i, u, v);
        
        ProjectedPoint & curr_pt = projected_cloud->points[i];
        normal_cloud->points[i].x = curr_pt.x;
        normal_cloud->points[i].y = curr_pt.y;
        normal_cloud->points[i].z = curr_pt.z;
        normal_cloud->points[i].intensity = curr_pt.intensity;
        normal_cloud->points[i].range = curr_pt.range;

        // Initialize classified cloud point
        classified_cloud->points[i].x = curr_pt.x;
        classified_cloud->points[i].y = curr_pt.y;
        classified_cloud->points[i].z = curr_pt.z;
        classified_cloud->points[i].intensity = curr_pt.intensity;
        classified_cloud->points[i].range = curr_pt.range;
        classified_cloud->points[i].normal_x = 0.0f;
        classified_cloud->points[i].normal_y = 0.0f;
        classified_cloud->points[i].normal_z = 0.0f;
        classified_cloud->points[i].valid = 0;
        classified_cloud->points[i].classification = UNKNOWN;
        classified_cloud->points[i].rgb = 0;  // Black color initially

        if(curr_pt.valid == 0) {
            continue;
        }

        if(u + normal_neighbor >= hor_pixel_num || (u-normal_neighbor < 0) ||
           v + normal_neighbor >= ver_pixel_num || (v-normal_neighbor < 0)) {
            continue;
        }

        double dzdpsi_sum = 0.0;
        double dzdtheta_sum = 0.0;

        int dpsi_sample_no = 0;
        int dtheta_sample_no = 0;

        for(int j = -normal_neighbor; j <= normal_neighbor; ++j) {
            for(int k = -normal_neighbor; k <= normal_neighbor; ++k) {
                int query_i = uv2index(u+j, v+k);
                ProjectedPoint & query_pt = projected_cloud->points[query_i];
                if(query_pt.valid == 0) {
                    continue;
                }
                // if(std::abs(query_pt.range - curr_pt.range) > 1.5) {
                //     continue;
                // }
                //Horizontal
                for(int l = j+1; l <= normal_neighbor; ++l) {
                    int target_i = uv2index(u+l, v+k);
                    ProjectedPoint & target_pt = projected_cloud->points[target_i];
                    if(target_pt.valid == 0) {
                        continue;
                    }
                    // if(std::abs(target_pt.range - query_pt.range) > 1.5) {
                    //     continue;
                    // }

                    dzdpsi_sum += (target_pt.range - query_pt.range)/(float(l-j)*hor_resolution*curr_pt.range);
                    ++dpsi_sample_no;
                }

                for(int l = k+1; l <= normal_neighbor; ++l) {
                    int target_i = uv2index(u+j, v+l);
                    ProjectedPoint & target_pt = projected_cloud->points[target_i];
                    if(target_pt.valid == 0) {
                        continue;
                    }
                    // if(std::abs(target_pt.range - query_pt.range) > 1.5) {
                    //     continue;
                    // }

                    dzdtheta_sum += (target_pt.range - query_pt.range)/(float(l-k)*ver_resolution*curr_pt.range);
                    ++dtheta_sample_no;
                }
            }
        }

        if(dpsi_sample_no < normal_neighbor*2 || dtheta_sample_no < normal_neighbor*2) {
            continue;
        }
        float dzdpsi_mean = dzdpsi_sum/float(dpsi_sample_no);
        float dzdtheta_mean = dzdtheta_sum/float(dtheta_sample_no);

        Eigen::Vector3f normal_sp{dzdpsi_mean, -dzdtheta_mean, 1};
        normal_sp.normalize();

        Eigen::Vector3f ray_dir{curr_pt.x, curr_pt.y, curr_pt.z};
        ray_dir.normalize();

        Eigen::Vector3f normal = sp2cart_map[i]*normal_sp;
        normal.normalize();

        if(normal.dot(ray_dir) > 0) {
            normal = -normal;
        }

        float d = -(normal.x()*curr_pt.x + normal.y()*curr_pt.y + normal.z()*curr_pt.z);

        bool valid_normal = true;;

        int valid_neighbors = 0;

        for(int j = -normal_neighbor; j <= normal_neighbor; ++j) {
            for(int k = -normal_neighbor; k <= normal_neighbor; ++k) {
                int query_i = uv2index(u+j, v+k);
                ProjectedPoint & target_pt = projected_cloud->points[query_i];
                float dist = std::abs(d + normal.x()*target_pt.x + normal.y() * target_pt.y + normal.z() *target_pt.z);

                if(dist < 0.15) {
                    ++valid_neighbors;
                }

            }
        }
        if(valid_neighbors < (2*normal_neighbor+1)*(2*normal_neighbor+1)/3) {
            continue;
        }

        normal_cloud->points[i].valid = 1;
        normal_cloud->points[i].normal_x = normal.x();
        normal_cloud->points[i].normal_y = normal.y();
        normal_cloud->points[i].normal_z = normal.z();

        // Also populate classified cloud
        classified_cloud->points[i].x = curr_pt.x;
        classified_cloud->points[i].y = curr_pt.y;
        classified_cloud->points[i].z = curr_pt.z;
        classified_cloud->points[i].intensity = curr_pt.intensity;
        classified_cloud->points[i].range = curr_pt.range;
        classified_cloud->points[i].normal_x = normal.x();
        classified_cloud->points[i].normal_y = normal.y();
        classified_cloud->points[i].normal_z = normal.z();
        classified_cloud->points[i].valid = 1;
        classified_cloud->points[i].classification = UNKNOWN; // Will be updated in classification step

        if(show_img == 1) {
            int r = 0;
            int g = 0;
            int b = 0;
            r = int((normal(0)*0.5+0.5) * 255);
            g = int((normal(1)*0.5+0.5) * 255);
            b = int((normal(2)*0.5+0.5) * 255);
            normal_img.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);

            int r_sp = 0;
            int g_sp = 0;
            int b_sp = 0;
            r_sp = int((normal_sp(0)*0.5+0.5) * 255);
            g_sp = int((normal_sp(1)*0.5+0.5) * 255);
            b_sp = int((normal_sp(2)*0.5+0.5) * 255);


            normal_img_sp.at<cv::Vec3b>(v, u) = cv::Vec3b(b_sp, g_sp, r_sp);
        }


    }
    
    // Perform classification after normal extraction
    if(show_img == 1) {
        int ground_count = 0, roof_count = 0, wall_count = 0, edge_count = 0, unknown_count = 0;
        
        #pragma omp parallel for num_threads(4) reduction(+:ground_count,roof_count,wall_count,edge_count,unknown_count)
        for(int i = 0; i < projected_cloud->size(); ++i) {
            int u, v;
            index2uv(i, u, v);
            
            PointClass classification = classifyPixel(u, v, normal_cloud);
            
            // Get BGR color for image display
            cv::Vec3b color_bgr = getClassificationColor(classification);
            classification_img.at<cv::Vec3b>(v, u) = color_bgr;
            
            // Update classified point cloud with classification and RGB color
            classified_cloud->points[i].classification = static_cast<uint8_t>(classification);
            classified_cloud->points[i].rgb = getClassificationColorRGB(classification);
            
            // Count classifications
            switch(classification) {
                case GROUND: ground_count++; break;
                case ROOF: roof_count++; break;
                case WALL: wall_count++; break;
                case EDGE: edge_count++; break;
                default: unknown_count++; break;
            }
        }
        
        // Print classification statistics
        int total_classified = ground_count + roof_count + wall_count + edge_count;
        if(total_classified > 0) {
            ROS_INFO("Classification: Ground:%d(%.1f%%) Roof:%d(%.1f%%) Wall:%d(%.1f%%) Edge:%d(%.1f%%) Unknown:%d", 
                ground_count, 100.0*ground_count/total_classified,
                roof_count, 100.0*roof_count/total_classified,
                wall_count, 100.0*wall_count/total_classified,
                edge_count, 100.0*edge_count/total_classified,
                unknown_count);
        }
    } else {
        // Even if show_img is off, still perform classification for classified cloud
        #pragma omp parallel for num_threads(4)
        for(int i = 0; i < projected_cloud->size(); ++i) {
            int u, v;
            index2uv(i, u, v);
            
            PointClass classification = classifyPixel(u, v, normal_cloud);
            classified_cloud->points[i].classification = static_cast<uint8_t>(classification);
            classified_cloud->points[i].rgb = getClassificationColorRGB(classification);
        }
    }
    
    if(show_img==1) {
        // Stack all three images vertically: normal, normal_sp, classification
        cv::Mat combined_img;
        cv::vconcat(normal_img, normal_img_sp, combined_img);
        cv::vconcat(combined_img, classification_img, combined_img);
        cv::imshow("Normal + Classification", combined_img);
        cv::waitKey(1);
    }

}

void OnSubscribeProjectedPointCloud(const sensor_msgs::PointCloud2ConstPtr & msg) {
    pcl::PointCloud<ProjectedPoint>::Ptr projected_cloud(new pcl::PointCloud<ProjectedPoint>);
    pcl::fromROSMsg(*msg, *projected_cloud);
    pcl::PointCloud<PointsWithNormals>::Ptr normal_cloud(new pcl::PointCloud<PointsWithNormals>(hor_pixel_num, ver_pixel_num));
    pcl::PointCloud<ClassifiedPoint>::Ptr classified_cloud(new pcl::PointCloud<ClassifiedPoint>(hor_pixel_num, ver_pixel_num));
    
    NormalExtraction(projected_cloud, normal_cloud, classified_cloud);

    // Filter out unclassified points and assign colors from nearest classified neighbors
    pcl::PointCloud<PointsWithNormals>::Ptr filtered_normal_cloud(new pcl::PointCloud<PointsWithNormals>);
    pcl::PointCloud<ClassifiedPoint>::Ptr filtered_classified_cloud(new pcl::PointCloud<ClassifiedPoint>);
    
    // Individual classification clouds
    pcl::PointCloud<ClassifiedPoint>::Ptr ground_cloud(new pcl::PointCloud<ClassifiedPoint>);
    pcl::PointCloud<ClassifiedPoint>::Ptr roof_cloud(new pcl::PointCloud<ClassifiedPoint>);
    pcl::PointCloud<ClassifiedPoint>::Ptr wall_cloud(new pcl::PointCloud<ClassifiedPoint>);
    pcl::PointCloud<ClassifiedPoint>::Ptr edge_cloud(new pcl::PointCloud<ClassifiedPoint>);
    pcl::PointCloud<ClassifiedPoint>::Ptr non_planar_cloud(new pcl::PointCloud<ClassifiedPoint>);
    
    // First pass: collect all classified points and separate by type
    int classified_count = 0;
    for(int i = 0; i < classified_cloud->size(); ++i) {
        ClassifiedPoint& pt = classified_cloud->points[i];
        if(pt.valid == 1 && pt.classification != UNKNOWN) {
            filtered_normal_cloud->push_back(normal_cloud->points[i]);
            filtered_classified_cloud->push_back(pt);
            classified_count++;
            
            // Add to specific classification clouds
            switch(pt.classification) {
                case GROUND:
                    ground_cloud->push_back(pt);
                    break;
                case ROOF:
                    roof_cloud->push_back(pt);
                    break;
                case WALL:
                    wall_cloud->push_back(pt);
                    break;
                case EDGE:
                    edge_cloud->push_back(pt);
                    break;
            }
        }
    }
    
    ROS_INFO("First pass: Found %d classified points out of %d total points", classified_count, (int)classified_cloud->size());
    
    // Collect non-planar points (valid original points but invalid normals)
    int non_planar_count = 0;
    for(int i = 0; i < classified_cloud->size(); ++i) {
        ProjectedPoint& orig_pt = projected_cloud->points[i];
        ClassifiedPoint& pt = classified_cloud->points[i];
        
        // If original point was valid but normal extraction failed (valid == 0)
        if(orig_pt.valid == 1 && pt.valid == 0) {
            // Create a point with invalid normal but valid position
            ClassifiedPoint non_planar_pt;
            non_planar_pt.x = orig_pt.x;
            non_planar_pt.y = orig_pt.y;
            non_planar_pt.z = orig_pt.z;
            non_planar_pt.intensity = orig_pt.intensity;
            non_planar_pt.range = orig_pt.range;
            non_planar_pt.normal_x = 0.0f;
            non_planar_pt.normal_y = 0.0f;
            non_planar_pt.normal_z = 0.0f;
            non_planar_pt.valid = 0;  // Mark as invalid normal
            non_planar_pt.classification = UNKNOWN;
            non_planar_pt.rgb = 0x808080;  // Gray color for non-planar points
            
            non_planar_cloud->push_back(non_planar_pt);
            non_planar_count++;
        }
    }
    
    ROS_INFO("Found %d non-planar points", non_planar_count);
    
    // Second pass: handle unclassified points by finding nearest classified neighbor
    for(int i = 0; i < classified_cloud->size(); ++i) {
        ClassifiedPoint& pt = classified_cloud->points[i];
        if(pt.valid == 1 && pt.classification == UNKNOWN) {
            int u, v;
            index2uv(i, u, v);
            
            // Search for nearest classified point in expanding radius
            uint32_t nearest_color = 0;
            uint8_t nearest_classification = UNKNOWN;
            bool found_neighbor = false;
            
            for(int radius = 1; radius <= 10 && !found_neighbor; ++radius) {
                for(int j = -radius; j <= radius && !found_neighbor; ++j) {
                    for(int k = -radius; k <= radius && !found_neighbor; ++k) {
                        if(abs(j) != radius && abs(k) != radius) continue; // Only check perimeter
                        
                        int neighbor_u = u + j;
                        int neighbor_v = v + k;
                        
                        if(neighbor_u >= 0 && neighbor_u < hor_pixel_num && 
                           neighbor_v >= 0 && neighbor_v < ver_pixel_num) {
                            int neighbor_i = uv2index(neighbor_u, neighbor_v);
                            if(neighbor_i < classified_cloud->size()) {
                                ClassifiedPoint& neighbor_pt = classified_cloud->points[neighbor_i];
                                if(neighbor_pt.valid == 1 && neighbor_pt.classification != UNKNOWN) {
                                    nearest_color = neighbor_pt.rgb;
                                    nearest_classification = neighbor_pt.classification;
                                    found_neighbor = true;
                                }
                            }
                        }
                    }
                }
            }
            
            if(found_neighbor) {
                pt.rgb = nearest_color;
                pt.classification = nearest_classification;
                filtered_normal_cloud->push_back(normal_cloud->points[i]);
                filtered_classified_cloud->push_back(pt);
                
                // Add to specific classification clouds based on inherited classification
                switch(pt.classification) {
                    case GROUND:
                        ground_cloud->push_back(pt);
                        break;
                    case ROOF:
                        roof_cloud->push_back(pt);
                        break;
                    case WALL:
                        wall_cloud->push_back(pt);
                        break;
                    case EDGE:
                        edge_cloud->push_back(pt);
                        break;
                }
            }
        }
    }

    // Publish filtered normal point cloud
    sensor_msgs::PointCloud2 normal_cloud_msg;
    pcl::toROSMsg(*filtered_normal_cloud, normal_cloud_msg);
    normal_cloud_msg.header = msg->header;
    pubNormalPointCloud.publish(normal_cloud_msg);

    // Publish filtered classified point cloud
    sensor_msgs::PointCloud2 classified_cloud_msg;
    pcl::toROSMsg(*filtered_classified_cloud, classified_cloud_msg);
    classified_cloud_msg.header = msg->header;
    pubClassifiedPointCloud.publish(classified_cloud_msg);
    
    // Publish individual classification clouds
    sensor_msgs::PointCloud2 ground_cloud_msg;
    pcl::toROSMsg(*ground_cloud, ground_cloud_msg);
    ground_cloud_msg.header = msg->header;
    pubGroundPointCloud.publish(ground_cloud_msg);
    
    sensor_msgs::PointCloud2 roof_cloud_msg;
    pcl::toROSMsg(*roof_cloud, roof_cloud_msg);
    roof_cloud_msg.header = msg->header;
    pubRoofPointCloud.publish(roof_cloud_msg);
    
    sensor_msgs::PointCloud2 wall_cloud_msg;
    pcl::toROSMsg(*wall_cloud, wall_cloud_msg);
    wall_cloud_msg.header = msg->header;
    pubWallPointCloud.publish(wall_cloud_msg);
    
    sensor_msgs::PointCloud2 edge_cloud_msg;
    pcl::toROSMsg(*edge_cloud, edge_cloud_msg);
    edge_cloud_msg.header = msg->header;
    pubEdgePointCloud.publish(edge_cloud_msg);
    
    // Publish non-planar point cloud
    sensor_msgs::PointCloud2 non_planar_cloud_msg;
    pcl::toROSMsg(*non_planar_cloud, non_planar_cloud_msg);
    non_planar_cloud_msg.header = msg->header;
    pubNonPlanarPointCloud.publish(non_planar_cloud_msg);
}


int main(int argc, char ** argv) {

    ros::init(argc, argv, "nv_normal_extraction_node");
    ros::NodeHandle nh;

    nh.param<int>("nv_liom/horizontal_pixel_num", hor_pixel_num, 1024);
    nh.param<int>("nv_liom/vertical_pixel_num", ver_pixel_num, 64);
    nh.param<float>("nv_liom/horizontal_fov", hor_fov, 360.0);
    nh.param<float>("nv_liom/vertical_max", ver_max, 22.5);
    nh.param<float>("nv_liom/vertical_min", ver_min, -22.5);

    nh.param<float>("nv_liom/minimum_distance", min_dist, 1.0);
    nh.param<float>("nv_liom/maximum_distance", max_dist, 200.0);
    nh.param<int>("nv_liom/show_img", show_img, 1);
    nh.param<std::string>("nv_liom/mapping_save_dir", map_save_dir, "/home/morin/map");
    nh.param<int>("nv_liom/normal_neighbor", normal_neighbor, 2);
    nh.param<float>("nv_liom/dominance_threshold", dominance_threshold, 0.85);
    nh.param<float>("nv_liom/angle_variance_threshold_deg", angle_variance_threshold, 30.0);
    
    // Convert angle threshold from degrees to radians
    angle_variance_threshold = angle_variance_threshold * M_PI / 180.0f;

    hor_resolution = (hor_fov * M_PI/180.0f)/float(hor_pixel_num);
    ver_resolution = ((ver_max-ver_min) * M_PI/180.0f)/float(ver_pixel_num);

    OnInitialization();

    ros::Subscriber subProjectedPoints = nh.subscribe<sensor_msgs::PointCloud2>("/nv_liom/projected_cloud", 100, OnSubscribeProjectedPointCloud);
    pubNormalPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/normal_vector_cloud", 1000);
    pubClassifiedPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/classified_cloud", 1000);
    pubGroundPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/ground_cloud", 1000);
    pubRoofPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/roof_cloud", 1000);
    pubWallPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/wall_cloud", 1000);
    pubEdgePointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/edge_cloud", 1000);
    pubNonPlanarPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/nv_liom/non_planar_cloud", 1000);

    ROS_INFO("\033[1;34mLiDAR Normal Extraction Node Started\033[0m");

    ros::spin();

    return 0;
}