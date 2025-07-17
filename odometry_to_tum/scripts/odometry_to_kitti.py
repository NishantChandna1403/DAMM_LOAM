#!/usr/bin/env python3

import rospy
import os
from datetime import datetime
from nav_msgs.msg import Odometry
import numpy as np

class OdometryToKITTI:
    def __init__(self):
        rospy.init_node('odometry_to_kitti', anonymous=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create specific directory path
        save_dir = '/root/catkin_ws/src/dtc_ugv/odometry_to_kitti'
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(save_dir, exist_ok=True)
            rospy.loginfo(f"Save directory: {save_dir}")
        except Exception as e:
            rospy.logerr(f"Failed to create directory {save_dir}: {e}")
            rospy.signal_shutdown("Directory creation failed")
            return
        
        # Create full file path
        self.filename = os.path.join(save_dir, f'trajectory_kitti_{timestamp}.txt')
        
        # Try to create and open the file
        try:
            self.file = open(self.filename, 'w')
            rospy.loginfo(f"Successfully created file: {os.path.abspath(self.filename)}")
        except Exception as e:
            rospy.logerr(f"Failed to create file {self.filename}: {e}")
            rospy.signal_shutdown("File creation failed")
            return
        
        self.msg_count = 0
        self.last_msg_time = rospy.Time.now()
        
        # Create subscriber
        self.subscription = rospy.Subscriber(
            '/genz/odometry',
            Odometry,
            self.listener_callback,
            queue_size=10
        )
        
        # Create timer to check for data timeout (check every 2 seconds)
        self.timeout_timer = rospy.Timer(rospy.Duration(2.0), self.check_timeout)
        
        rospy.loginfo("Odometry to KITTI converter started. Waiting for messages...")
        rospy.loginfo("Node will shutdown if no data received for 10 seconds")

    def listener_callback(self, msg):
        try:
            # Update last message time
            self.last_msg_time = rospy.Time.now()
            
            # Extract translation
            tx = msg.pose.pose.position.x
            ty = msg.pose.pose.position.y
            tz = msg.pose.pose.position.z

            # Extract orientation quaternion
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # Convert quaternion to rotation matrix
            r = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

            # Create the 3x4 transformation matrix for KITTI format
            # [R | t]
            # The KITTI format is the flattened version of this 3x4 matrix
            # r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
            kitti_matrix = [
                r[0,0], r[0,1], r[0,2], tx,
                r[1,0], r[1,1], r[1,2], ty,
                r[2,0], r[2,1], r[2,2], tz
            ]

            # Write to file
            line = " ".join([f"{val:.6f}" for val in kitti_matrix]) + "\n"
            self.file.write(line)
            self.file.flush()  # Force write to disk immediately
            
            self.msg_count += 1
            
            # Log progress every 50 messages
            if self.msg_count == 1:
                rospy.loginfo("First message received and recorded!")
            elif self.msg_count % 50 == 0:
                rospy.loginfo(f"Recorded {self.msg_count} trajectory points")
                
        except Exception as e:
            rospy.logerr(f"Error processing odometry message: {e}")

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """Convert a quaternion into a 3x3 rotation matrix."""
        x, y, z, w = qx, qy, qz, qw
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, xw = x*y, x*z, x*w
        yz, yw, zw = y*z, y*w, z*w

        return np.array([
            [1 - 2*yy - 2*zz, 2*xy - 2*zw,     2*xz + 2*yw],
            [2*xy + 2*zw,     1 - 2*xx - 2*zz, 2*yz - 2*xw],
            [2*xz - 2*yw,     2*yz + 2*xw,     1 - 2*xx - 2*yy]
        ])

    def check_timeout(self, event):
        """Check if we haven't received data for more than 10 seconds"""
        current_time = rospy.Time.now()
        time_since_last_msg = (current_time - self.last_msg_time).to_sec()
        
        if time_since_last_msg > 10.0:
            rospy.logwarn(f"No odometry data received for {time_since_last_msg:.1f} seconds")
            rospy.logwarn("Shutting down due to data timeout")
            rospy.signal_shutdown("Data timeout - no odometry messages for 10+ seconds")

    def shutdown_hook(self):
        """Clean shutdown - close file and print summary"""
        try:
            if hasattr(self, 'timeout_timer'):
                self.timeout_timer.shutdown()
            
            if hasattr(self, 'file') and self.file and not self.file.closed:
                self.file.close()
                
                # Check if file exists and has content
                if os.path.exists(self.filename):
                    file_size = os.path.getsize(self.filename)
                    rospy.loginfo(f"Trajectory saved successfully!")
                    rospy.loginfo(f"File: {os.path.abspath(self.filename)}")
                    rospy.loginfo(f"Total messages recorded: {self.msg_count}")
                    rospy.loginfo(f"File size: {file_size} bytes")
                    
                    if file_size == 0:
                        rospy.logwarn("Warning: File is empty - no data was recorded")
                else:
                    rospy.logerr("Error: File was not created properly")
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")


def main():
    try:
        node = OdometryToKITTI()
        rospy.on_shutdown(node.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
