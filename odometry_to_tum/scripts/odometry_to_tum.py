#!/usr/bin/env python3

import rospy
import os
from datetime import datetime
from nav_msgs.msg import Odometry


class OdometryToTUM:
    def __init__(self):
        rospy.init_node('odometry_to_tum', anonymous=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create specific directory path
        save_dir = '/root/catkin_ws/src/dtc_ugv/odometry_to_tum'
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(save_dir, exist_ok=True)
            rospy.loginfo(f"Save directory: {save_dir}")
        except Exception as e:
            rospy.logerr(f"Failed to create directory {save_dir}: {e}")
            rospy.signal_shutdown("Directory creation failed")
            return
        
        # Create full file path
        self.filename = os.path.join(save_dir, f'trajectory_tum_{timestamp}.txt')
        
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
        
        rospy.loginfo("Odometry to TUM converter started. Waiting for messages...")
        rospy.loginfo("Node will shutdown if no data received for 10 seconds")

    def listener_callback(self, msg):
        try:
            # Update last message time
            self.last_msg_time = rospy.Time.now()
            
            # Extract data
            timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            tx = msg.pose.pose.position.x
            ty = msg.pose.pose.position.y
            tz = msg.pose.pose.position.z
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # Write to file
            line = f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            self.file.write(line)
            self.file.flush()  # Force write to disk immediately
            
            self.msg_count += 1
            
            # Log progress every 50 messages instead of every 5 seconds
            if self.msg_count == 1:
                rospy.loginfo("First message received and recorded!")
            elif self.msg_count % 50 == 0:
                rospy.loginfo(f"Recorded {self.msg_count} trajectory points")
                
        except Exception as e:
            rospy.logerr(f"Error processing odometry message: {e}")

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
        node = OdometryToTUM()
        rospy.on_shutdown(node.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()