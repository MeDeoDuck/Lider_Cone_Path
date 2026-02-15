#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>

ros::Publisher pose_pub;

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    // PoseStamped 메시지 생성
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = msg->header.stamp; // PointCloud2 메시지의 타임스탬프 사용
    pose_msg.header.frame_id = "base_link";

    // Pose 데이터 설정 (필요한 데이터로 변경 가능)
    pose_msg.pose.position.x = 0.0;
    pose_msg.pose.position.y = 0.0;
    pose_msg.pose.position.z = 0.0;
    pose_msg.pose.orientation.x = 0.0;
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = 0.0;
    pose_msg.pose.orientation.w = 1.0;

    // 메시지 publish
    pose_pub.publish(pose_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "GPS_maker");
    ros::NodeHandle nh;

    // Publisher 설정
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/local_msgs_to_vision", 10);

    // Subscriber 설정
    ros::Subscriber pointcloud_sub = nh.subscribe("/os_cloud_node/points", 10, pointCloudCallback);
    ros::Subscriber pointcloud_sub_ = nh.subscribe("/scan_2D", 10, pointCloudCallback);


    // ROS 루프 시작
    ros::spin();

    return 0;
}
