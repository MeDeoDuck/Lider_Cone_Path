#include "pathPlaner.h"

// 척력 계산 함수
double APFPlanner::computeForce(double distance, double forceThreshold) {
    if (distance < 3.0) {
        return std::pow(distance - 3.0, 2) * (distance + 1.0);
    } else if (distance < forceThreshold) {
        return distance - 3.0;
    } else {
        return forceThreshold - 3.0;
    }
}

// 인력 계산 함수
double APFPlanner::computeAttractiveForce(double point_x, double point_y, double ego_x, double ego_y, double heading_angle, double max_distance) {
    double line_end_x = ego_x + max_distance * std::cos(heading_angle);
    double line_end_y = ego_y + max_distance * std::sin(heading_angle);

    double dx = line_end_x - ego_x;
    double dy = line_end_y - ego_y;
    double line_length = std::sqrt(dx * dx + dy * dy);

    if (line_length == 0.0) {
        return 0.0;
    }

    double projection = ((point_x - ego_x) * dx + (point_y - ego_y) * dy) / line_length;
    if (projection < 0.0 || projection > line_length) {
        return 0.0;
    }

    double closest_x = ego_x + projection * dx / line_length;
    double closest_y = ego_y + projection * dy / line_length;
    double distance_to_line = std::sqrt(std::pow(point_x - closest_x, 2) + std::pow(point_y - closest_y, 2));

    double attractive_force = -1.0 / 5.0 * std::pow(distance_to_line, 2) + 0.7;
    return attractive_force;
}

// 반발력 계산 함수
double APFPlanner::computeRepulsionAtPoint(double point_x, double point_y, const pcl::PointCloud<PointType>::Ptr& obstacles) {
    double total_repulsion = 0.0;
    for (const auto& obs : obstacles->points) {
        double distance = std::sqrt(std::pow(point_x - obs.x, 2) + std::pow(point_y - obs.y, 2));
        total_repulsion += computeForce(distance, FORCE_THRESHOLD);
    }
    return total_repulsion;
}

// 각 거리에서 최소 반발력 각도 구하기
std::vector<double> APFPlanner::findMinForceAngles(double ego_x, double ego_y, double ego_heading, const pcl::PointCloud<PointType>::Ptr& obstacles) {
    std::vector<double> min_angles;

    for (double distance : DISTANCES) {
        double min_force = std::numeric_limits<double>::max();
        double min_angle = 0.0;

        for (double angle : ANGLE_RANGE) {
            double theta = ego_heading + angle * M_PI / 180.0;
            double point_x = ego_x + distance * std::cos(theta);
            double point_y = ego_y + distance * std::sin(theta);

            double repulsion_force = computeRepulsionAtPoint(point_x, point_y, obstacles);
            double attractive_force = computeAttractiveForce(point_x, point_y, ego_x, ego_y, ego_heading);
            double total_force = repulsion_force - attractive_force;

            if (total_force < min_force) {
                min_force = total_force;
                min_angle = theta;
            }
        }

        min_angles.push_back(min_angle);
    }

    return min_angles;
}

// APF 기반 방향 계산 함수 (한 번의 업데이트만 수행)
void APFPlanner::getAPFAngle(const pcl::PointCloud<PointType>::Ptr& input_pointcloud, std::shared_ptr<visualization_msgs::MarkerArray> output_markerArray, int num_points, double step_distance, bool GPS2Lidar) {
    double current_x = 0;
    double current_y = 0;
    double current_heading = 0;

    pcl::PointCloud<PointType>::Ptr relative_pointcloud(new pcl::PointCloud<PointType>(*input_pointcloud));
    if (GPS2Lidar == true) {
        for (auto& point : relative_pointcloud->points) {
            point.x += -LI_TO_GPS_X;
        }
    }

    std::vector<double> min_angles = findMinForceAngles(current_x, current_y, current_heading, relative_pointcloud);

    double weighted_angle = 0.0;
    for (size_t i = 0; i < min_angles.size(); ++i) {
        weighted_angle += min_angles[i] * DISTANCE_WEIGHTS[i];
    }

    // 다음 포인트 계산
    double next_x = current_x + step_distance * std::cos(weighted_angle);
    double next_y = current_y + step_distance * std::sin(weighted_angle);

    // 마커 생성 및 추가
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = 0;

    geometry_msgs::Point start, end;
    start.x = current_x;
    start.y = current_y;
    start.z = 0.0;

    end.x = next_x;
    end.y = next_y;
    end.z = 0.0;

    marker.points.push_back(start);
    marker.points.push_back(end);
    marker.scale.x = 0.1;
    marker.scale.y = 0.2;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    output_markerArray->markers.push_back(marker);
}

