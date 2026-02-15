#include "preprocessor.h"


void Preprocessor::initCol(std::vector<std::vector<float>>& table, int col_num, int init_num)
{
    for (std::vector<float>& row : table)
    {
        if(col_num < row.size())
        {
            row[col_num] = init_num;
        }
    }
}


// Bounding Box 설정하는 코드
bool Preprocessor::marker_is_in_area(const std::shared_ptr<visualization_msgs::MarkerArray>& carBboxArray,
                                       const std::pair<double, double> x_threshold,
                                       const std::pair<double, double> y_threshold)
{
    std::cout << "testing!" << std::endl;
    // carBboxArray가 nullptr인지 먼저 확인
    if (!carBboxArray) return false;
    
    for (const auto& marker : carBboxArray->markers)
    {
        // Marker의 중심 좌표와 scale (크기)를 이용해 바운딩 박스 계산
        double center_x = marker.pose.position.x;
        double center_y = marker.pose.position.y;
        double half_width = marker.scale.x / 2.0;
        double half_height = marker.scale.y / 2.0;

        double min_x = center_x - half_width;
        double max_x = center_x + half_width;
        double min_y = center_y - half_height;
        double max_y = center_y + half_height;

        // x, y축 각각에서 영역이 겹치는지 확인
        if (min_x <= x_threshold.second && max_x >= x_threshold.first &&
            min_y <= y_threshold.second && max_y >= y_threshold.first)
        {
            return true; // 겹치는 경우
        }
    }
    return false; // 아무 marker도 영역과 겹치지 않으면 false 반환
}



// void Preprocessor::createAveragePointCloud(const std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>>& smallObject_Cloud_vector,
//                                            const pcl::PointCloud<PointType>::Ptr &avgCloudOut)
// {
//     // avgCloudOut을 새 포인트 클라우드로 초기화
//     avgCloudOut->clear();
    
//     // 각 클러스터에 대해 centroid를 계산하여 avgCloudOut에 추가
//     for (const auto& cloud : *smallObject_Cloud_vector)
//     {
//         if (cloud->points.empty())
//             continue;

//         PointType centroid;
//         centroid.x = 0.0;
//         centroid.y = 0.0;
//         centroid.z = 0.0;

//         // 클러스터 내의 모든 점의 좌표 합산
//         for (const auto& pt : cloud->points)
//         {
//             centroid.x += pt.x;
//             centroid.y += pt.y;
//             centroid.z += pt.z;
//         }
//         float num = static_cast<float>(cloud->points.size());
//         centroid.x /= num;
//         centroid.y /= num;
//         centroid.z /= num;

//         avgCloudOut->points.push_back(centroid);
//     }

//     // avgCloudOut의 메타 정보 설정
//     avgCloudOut->width = avgCloudOut->points.size();
//     avgCloudOut->height = 1;
//     avgCloudOut->is_dense = true;
// }



/**
 * @brief 몸체에 찍히는 포인트 클라우드를 제거한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
*/
// 차량 본체 지우는 코드
void Preprocessor::removeBody(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud)
{
    pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>);
    // X 축으로 filtering
    pcl::PassThrough<PointType> xfilter;
    pcl::PassThrough<PointType> yfilter;

    xfilter.setInputCloud(input_pointCloud);
    xfilter.setFilterFieldName("x");
    xfilter.setFilterLimits(-0.4, 1.5); // 뒤, 앞
    xfilter.setFilterLimitsNegative(true);
    xfilter.filter(*tempCloud);

    // 그 후 y축 방향으로 중앙에 있는 부분 제거
    yfilter.setInputCloud(input_pointCloud);
    yfilter.setFilterFieldName("y");
    yfilter.setFilterLimits(-0.6, 0.6);  // 좌, 우
    yfilter.setFilterLimitsNegative(true);
    yfilter.filter(*output_pointCloud);

    *output_pointCloud += *tempCloud;
}

// 라이다의 상대 좌표 GPS에 맞추기
/**
 * @brief 상대좌표 기준을 라이다에서 GPS로 이동한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
*/
void Preprocessor::calibrateLidar(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud, float Yangle)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(Yangle * M_PI / 180.0, Eigen::Vector3f::UnitY()));
    transform.translation() << LI_TO_GPS_X, 0.0, 0.0;
    pcl::transformPointCloud(*input_pointCloud, *output_pointCloud, transform);
}

// 센서 오작동으로 인해 잘못 검출한 데이터 제거 => 빛 반사, 하늘 등
/**
 * @brief 결측치를 제거한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
*/
void Preprocessor::removeNaN(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud)
{
    std::vector<int> _;
    pcl::removeNaNFromPointCloud(*input_pointCloud, *output_pointCloud, _);
}


// 지반 제거
/**
 * @brief 지면을 제거한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
 * @param z_threshold z값 범위
*/
void Preprocessor::removeFloor(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud, std::pair<double, double> z_threshold)
{
    pcl::PassThrough<PointType> passThrough;
    passThrough.setInputCloud(input_pointCloud);
    passThrough.setFilterFieldName("z");
    passThrough.setFilterLimits(z_threshold.first, z_threshold.second);
    passThrough.filter(*output_pointCloud);
}


/**
 * @brief voxel_grid로 다운셈플링한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
 * @param leaf_size voxel_grid할 규격
*/
void Preprocessor::voxelGride(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud, double leaf_size)
{
    pcl::VoxelGrid<PointType> voxelGrid;
    voxelGrid.setInputCloud(input_pointCloud);
    voxelGrid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxelGrid.filter(*output_pointCloud);
}


/**
 * @brief 3D 포인트 클라우드를 2D 포인트 클라우드로 압축한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
 * @param object_size 차원 압축시 2D voxel_grid할 규격
*/
void Preprocessor::convert3Dto2D(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud, double object_size)
{
    pcl::VoxelGrid<PointType> voxelGrid;
    voxelGrid.setInputCloud(input_pointCloud);
    voxelGrid.setLeafSize(object_size, object_size, object_size);
    voxelGrid.filter(*output_pointCloud);
    for (PointType& point : output_pointCloud->points)
    {
        point.z = 0.0;
        output_pointCloud->points.push_back(point);
    }
}


/**
 * @brief 모든 포인트의 중간값을 노이즈로 추가한다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
*/
void Preprocessor::addNoise(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, const pcl::PointCloud<PointType>::Ptr& output_pointCloud)
{
    for (size_t i = 0; i < input_pointCloud->size(); ++i)
    {
        for (size_t j = i+1; j < input_pointCloud->size(); ++j)
        {
            PointType midPoint;
            midPoint.x = ((*input_pointCloud)[i].x + (*input_pointCloud)[j].x) / 2.0f;
            midPoint.y = ((*input_pointCloud)[i].y + (*input_pointCloud)[j].y) / 2.0f;
            midPoint.z = ((*input_pointCloud)[i].z + (*input_pointCloud)[j].z) / 2.0f;

            output_pointCloud->push_back(midPoint);
        }
    }
}


/**
 * @brief x, y, z 범위 만큼 자른다.
 * @param input_point 자를 포인트 
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
 * @param xyz_threshold 포인트를 자를 범위 (최대값: -99999, 최소값: 99999)
*/
void Preprocessor::cutPointCloud(const pcl::PointCloud<PointType>::Ptr& input_pointCloud,
                                 const pcl::PointCloud<PointType>::Ptr& output_pointCloud,
                                 const std::pair<double, double>       x_threshold, 
                                 const std::pair<double, double>       y_threshold, 
                                 const std::pair<double, double>       z_threshold)
{
    for(const PointType& point : input_pointCloud->points){
        if(x_threshold.first < point.x && point.x < x_threshold.second &&
           y_threshold.first < point.y && point.y < y_threshold.second &&
           z_threshold.first < point.z && point.z < z_threshold.second)
        {
            output_pointCloud->points.push_back(point);
        }
    }
}


/**
 * @brief x, y, z 범위와 xy각도 만큼 자른다.
 * @param input_pointCloud 자를 포인트 클라우드
 * @param output_pointCloud 자른 포인트를 저장할 포인트 클라우드
 * @param xyz_threshold 포인트를 자를 범위 (최대값: -99999, 최소값: 99999)
 * @param xy_angle_threshold 포인트를 자를 xy 각도 (가운데로 부터 좌측: 음수, 우측: 양수)
*/
void Preprocessor::cutPointCloud(const pcl::PointCloud<PointType>::Ptr& input_pointCloud,
                                 const pcl::PointCloud<PointType>::Ptr& output_pointCloud,
                                 const std::pair<double, double>       x_threshold, 
                                 const std::pair<double, double>       y_threshold, 
                                 const std::pair<double, double>       z_threshold,
                                 const std::pair<float,  float >       xy_angle_threshold)
{
    for(const PointType& point : input_pointCloud->points){
        if (x_threshold.first < point.x && point.x < x_threshold.second &&
            y_threshold.first < point.y && point.y < y_threshold.second &&
            z_threshold.first < point.z && point.z < z_threshold.second)
        {
            double xy_angle = atan2(point.y, point.x) * RAD_TO_DEG;
            if (xy_angle_threshold.first < xy_angle && xy_angle < xy_angle_threshold.second)
            {
                output_pointCloud->points.push_back(point);
            }
        }
    }
}


/**
 * @brief pcl::PointCloud를 std::vector에 저장후 x값이 작은것부터 큰 순서로 정리한다.
 * @param input_pointCloud 자를 포인트 클라우드
 * @param output_vector 자른 포인트를 저장할 컨테이너
*/
void Preprocessor::convertPCLtoVector(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, std::unique_ptr<std::vector<PointType>>& output_vector) {
    // 포인트 클라우드의 모든 포인트를 벡터에 복사합니다.
    for (const auto& point : input_pointCloud->points) {
        output_vector->push_back(point);
    }

    // 벡터를 x 좌표에 따라 오름차순으로 정렬합니다.
    std::sort(output_vector->begin(), output_vector->end(), [](const PointType& a, const PointType& b) {
        return a.x < b.x;
    });
}


/**
 * @brief 2개의 점을 이은 선에 대해서 좌측, 우측, 위에 있는지 판단한다.
 * @param firstMidPoint 선이 될 첫번째 점
 * @param secondMidPoint 선이 될 두번째 점
 * @param point 확인할 대상
*/
float Preprocessor::crossLine(std::pair<float, float> firstMidPoint, std::pair<float, float> secondMidPoint, PointType point) {
    return (secondMidPoint.first - firstMidPoint.first) * (point.y - firstMidPoint.second) - (secondMidPoint.second - firstMidPoint.second) * (point.x - firstMidPoint.first);
}


bool Preprocessor::betweenLine(std::pair<std::pair<float, float>, std::pair<float, float>> pointArea, PointType point)
{
    // 매개변수 값 가져오기
    float x = pointArea.first.first;
    float y = pointArea.first.second;
    float distance = pointArea.second.first;
    float angle = pointArea.second.second;

    float x2 = x + cos(angle + PI/2) * distance;
    float y2 = y + sin(angle + PI/2) * distance;

    // 기울기
    float m = tan(angle);
    // 선 하나 생성
    float line1 = y + m * (point.x-x);

    // distance만큼 떨어진 선 하나 더 생성
    float line2 = y2 + m * (point.x-x2);

    if (line1 < point.y && point.y < line2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * 두 포인트 사이 거리를 리턴하되 제곱으로 리턴
*/
float Preprocessor::distanceTwoPointPower2(std::pair<float, float> midPoint, PointType point)
{
    return (std::pow(midPoint.first - point.x, 2) + std::pow(midPoint.second - point.y, 2));
}


bool Preprocessor::isWithinAngle(PointType point, std::pair<float, float> midPoint, float midPointAngle, float angle) 
{
    // 각도를 라디안으로 변환
    float radians = midPointAngle + angle * M_PI / 180.0;

    // 기울기 계산
    float slope = tan(radians);

    // y절편 계산
    float yIntercept = midPoint.second - slope * midPoint.first;

    // 점이 선의 위쪽에 있는지 확인
    bool isAbove = point.y >= -slope * point.x + yIntercept;

    // 점이 선의 아래쪽에 있는지 확인
    bool isBelow = point.y <=  slope * point.x + yIntercept;

    // 점이 선의 -60도와 60도 사이에 있는지 확인
    return isAbove && isBelow;
}

/**
 * (a, b)를 기준으로 기울기가 m인 직선과 m과 수직인 직선을 기준으로 4가지 영역을 분할해 (c,d)의 위치 리턴
*/
int determineRegion(double a, double b, double m, double c, double d) {
    bool L = (d - b) > m * (c - a); // 기준 직선에 대한 판별
    if      (L )  return 1; // 1번 영역
    else if (!L)  return 2; // 2번 영역
}