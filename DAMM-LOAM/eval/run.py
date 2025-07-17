#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry

class TumOdomLogger:
    def __init__(self):
        self.file = open("estimated.txt", "w", buffering=1)  # line-buffered
        self.file.write("# timestamp tx ty tz qx qy qz qw\n")
        rospy.Subscriber("/genz/odometry", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        timestamp = msg.header.stamp.to_sec()
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        line = f"{timestamp:.9f} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f} "
        line += f"{ori.x:.6f} {ori.y:.6f} {ori.z:.6f} {ori.w:.6f}\n"
        self.file.write(line)
        self.file.flush()  # flush after every write

    def shutdown(self):
        self.file.close()
        rospy.loginfo("✅ Saved estimated trajectory to estimated.txt")

if __name__ == "__main__":
    rospy.init_node("damm_odom_logger", anonymous=True)
    logger = TumOdomLogger()
    rospy.on_shutdown(logger.shutdown)
    rospy.loginfo("📡 Listening to /damm_loam/odometry and writing TUM format...")
    rospy.spin()

