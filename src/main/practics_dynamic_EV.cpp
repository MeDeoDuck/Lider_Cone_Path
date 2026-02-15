#include "pointcloud_generator.h"
#include "utility.h"
#include "pathPlaner.h"
#include "clustering.h"  // setConeROI() 함수가 정의된 헤더 포함
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>  // transformPointCloud 함수가 여기 있음

using namespace std;

/**
 ** 좌표
 * x: 차량의 정면
 * y: 차량의 좌측
 * z: 차량의 상측
 * 전방시야: x축으로부터 좌측이 음수, 우측이 양수 
 * 회전: 축을 기준으로 오른손의 법칙을 따름
 * 라이다의 row: 아래서부터 위로 숫자가 증가
 * 라이다의 column: 라이다 후면부터 반시계 방향으로 증가
 */

class ImageProjection {
private:
    // ros handle
    ros::NodeHandle nh;

    // 차량 좌표 및 시간
    geometry_msgs::PoseStamped local;

    // ros subscriber (velodyne_points만 수신)
    ros::Subscriber subLidarNormal;

    std::vector<message_filters::Subscriber<sensor_msgs::Image>*> subImages;
    std::vector<message_filters::Subscriber<vision_msgs::Detection2DArray>*> subBoxes;

    std::vector<std::vector<string>> camera_box_pairs;
    std::vector<std::vector<string>> camera_line_pairs;
    std::vector<G_CMD> g_cmd_obj_list;
    std::vector<G_CMD> g_cmd_line_list;
    std::vector<pid_t> pids;
    
    // ros publisher
    ros::Publisher pubOdometry;

    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;
    ros::Publisher pubInterestCloud;
    ros::Publisher pubInterestCloud2D;
    ros::Publisher pubROICloud;
    ros::Publisher pubFovCloud;
    ros::Publisher pubgroundCloud;
    ros::Publisher pubnongroundCloud;

    ros::Publisher pubTransformedCloud;
    ros::Publisher pubTransformedRoiCloud;
    ros::Publisher pubAngleCloud;

    ros::Publisher pubDebugCloud;
    ros::Publisher pubCarMarkerArray;    // 큐브
    ros::Publisher pubCarBboxArray;       // Bbox
    ros::Publisher pubObjMarkerArray;     // 작은 오브젝트 -> 콘, 배럴

    ros::Publisher pubConeClusterCloud;
    ros::Publisher pubAngleMarkerArray;

    ros::Publisher pubEmergencyFromLidar;
    ros::Publisher pubMidLineMarkerArray;  // New publisher for mid-line marker array



    // pcl pointCloud
    // 원본 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    // Publish할 pointCloud
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;
    pcl::PointCloud<PointType>::Ptr interestCloud;
    pcl::PointCloud<PointType>::Ptr interestCloud2D;
    pcl::PointCloud<PointType>::Ptr ROICloud;
    pcl::PointCloud<PointType>::Ptr fovCloud;
    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr nongroundCloud;
    pcl::PointCloud<PointType>::Ptr transformedCloud;
    pcl::PointCloud<PointType>::Ptr transformedRoiCloud;
    pcl::PointCloud<PointType>::Ptr debugCloud;
    pcl::PointCloud<PointType>::Ptr coneClusterCloud;
    pcl::PointCloud<PointType>::Ptr angleCloud;
  

    // Publish할 MarkerArray
    std::shared_ptr<visualization_msgs::MarkerArray> carMarkerArray = std::make_shared<visualization_msgs::MarkerArray>();
    std::shared_ptr<visualization_msgs::MarkerArray> carBboxArray = std::make_shared<visualization_msgs::MarkerArray>();
    std::shared_ptr<visualization_msgs::MarkerArray> objMarkerArray = std::make_shared<visualization_msgs::MarkerArray>();
    std::shared_ptr<visualization_msgs::MarkerArray> angleBboxArray = std::make_shared<visualization_msgs::MarkerArray>();
    std::shared_ptr<visualization_msgs::MarkerArray> angleMarkerArray = std::make_shared<visualization_msgs::MarkerArray>();
    std::shared_ptr<visualization_msgs::MarkerArray> midLineMarkerArray = std::make_shared<visualization_msgs::MarkerArray>();  // New marker array for center line



    // 클러스터를 담은 vector
    pcl::PointCloud<PointType>::Ptr smallObject_Cloud;
    std::shared_ptr<std::vector<pcl::PointCloud<PointType>::Ptr>> bigObject_Cloud_vector = std::make_shared<std::vector<pcl::PointCloud<PointType>::Ptr>>();
    
    std_msgs::Bool emergency;

    // ── 추가: 긴급정지 debounce 관련 변수 ──
    bool stable_emergency_state = false;
    int emergency_counter = 0;

    Preprocessor preprocessor;
    PointCloudGenerator pointCloudGenerator;
    APFPlanner pathPlaner;
    Clustering clustering;  // Added clustering instance
    

    // 소모 시간 측정을 위한 변수
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;


public:
    // 생성자: 초기화 및 토픽/프로세스 설정
    ImageProjection() : nh("~") {
        cout << "ImageProjection Start" << endl;
        // Publish할 토픽 설정
        pubOdometry = nh.advertise<geometry_msgs::PoseStamped>("/local_msg", 10);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

        pubInterestCloud = nh.advertise<sensor_msgs::PointCloud2>("/interest_cloud", 1);
        pubInterestCloud2D = nh.advertise<sensor_msgs::PointCloud2>("/interest_cloud2d", 1);
        pubROICloud = nh.advertise<sensor_msgs::PointCloud2>("/roi_cloud", 1);
        pubFovCloud = nh.advertise<sensor_msgs::PointCloud2>("/fov_cloud", 1);
        pubgroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubnongroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/nonground_cloud", 1);
        pubDebugCloud = nh.advertise<sensor_msgs::PointCloud2>("/debug_cloud", 1);
        pubTransformedCloud = nh.advertise<sensor_msgs::PointCloud2>("/transformed_cloud", 1);
        pubTransformedRoiCloud = nh.advertise<sensor_msgs::PointCloud2>("/transformed_roi_cloud", 1);
        pubConeClusterCloud = nh.advertise<sensor_msgs::PointCloud2>("/coneClusterCloud", 1);
        pubAngleCloud = nh.advertise<sensor_msgs::PointCloud2>("/angle_cloud", 1);

        pubCarMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("/markers_detected", 1);
        pubCarBboxArray = nh.advertise<visualization_msgs::MarkerArray>("/markers_vis", 1);
        pubObjMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("/obj_marker_array", 1);

        pubAngleMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("/angle_marker_array", 1);
        pubMidLineMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("/mid_line_marker_array", 1);  // New publisher
        pubEmergencyFromLidar = nh.advertise<std_msgs::Bool>("/emergency_from_lidar", 10);

        // 기존 동기화 방식 제거하고 velodyne_points 토픽만 subscribe
        subLidarNormal = nh.subscribe("/os_cloud_node/points", 10, &ImageProjection::cloudHandler, this);
        midLineMarkerArray = std::make_shared<visualization_msgs::MarkerArray>();  // Reinitialize

        mapReader("/home/kim/catkin_ws_practice/src/Gigacha_Lidar/json_map/real_real_final_map.json");
        allocateMemory();
        clearMemory();
    }

    ~ImageProjection() {
        // 자식 프로세스 종료 대기
        for (pid_t pid : pids) {
            int status;
            waitpid(pid, &status, 0);
        }
    }

    void allocateMemory() {
        cout << "allocate Memory start" << endl;
        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());
        interestCloud.reset(new pcl::PointCloud<PointType>());
        interestCloud2D.reset(new pcl::PointCloud<PointType>());
        ROICloud.reset(new pcl::PointCloud<PointType>());
        fovCloud.reset(new pcl::PointCloud<PointType>());
        groundCloud.reset(new pcl::PointCloud<PointType>());
        nongroundCloud.reset(new pcl::PointCloud<PointType>());
        debugCloud.reset(new pcl::PointCloud<PointType>());
        transformedCloud.reset(new pcl::PointCloud<PointType>());
        transformedRoiCloud.reset(new pcl::PointCloud<PointType>());
        debugCloud.reset(new pcl::PointCloud<PointType>());
        angleCloud.reset(new pcl::PointCloud<PointType>());
        coneClusterCloud.reset(new pcl::PointCloud<PointType>());
        smallObject_Cloud.reset(new pcl::PointCloud<PointType>());  
    }
    

    void clearMemory() {
        laserCloudIn->clear();

        fullCloud->clear();
        fullInfoCloud->clear();
        interestCloud->clear();
        interestCloud2D->clear();
        ROICloud->clear();
        fovCloud->clear();
        groundCloud->clear();
        nongroundCloud->clear();
        debugCloud->clear();
        transformedCloud->clear();
        transformedRoiCloud->clear();
        coneClusterCloud->clear();
        angleCloud->clear();

        // markerArray 초기화
        deleteAllMarkers(carMarkerArray);
        deleteAllMarkers(carBboxArray);
        deleteAllMarkers(objMarkerArray);
        deleteAllMarkers(angleMarkerArray);

        // vector 초기화
        smallObject_Cloud->clear();
        bigObject_Cloud_vector->clear();
    }

    void report() {
        cout << "\033[2J\033[H";
        cout << "===================================" << endl;
        cout << "              Process              " << endl;
        cout << "-----------------------------------" << endl;
        cout << " Big Object ----------- {" << bigObject_Cloud_vector->size() << "}" << endl;
        if (emergency.data)
            std::cout << "Emergency Stop --------- {Stop!!!}" << std::endl;
        else
            std::cout << "Emergency Stop --------- {Go!!!}" << std::endl;
        cout << "===================================" << endl;
        cout << "               Time                " << endl;
        cout << "-----------------------------------" << endl;
        cout << " Total ---------------- " 
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
             << "ms" << endl;
        cout << "===================================" << endl;
    }

    // velodyne_points 토픽만 수신 시 callback 실행
    void cloudHandler(const sensor_msgs::PointCloud2::ConstPtr& rosCloud) {
        // 라이다 메시지를 PCL 포인트클라우드로 변환
        pcl::fromROSMsg(*rosCloud, *laserCloudIn);
        
        // 현재 시간 저장
        saveCurrentTime(rosCloud);
        // 라이다-GPS 거리 및 라이다 Y축 회전값 보정
        preprocessor.calibrateLidar(laserCloudIn, laserCloudIn, 0);
        // 몸체에 찍히는 라이다 포인트 제거
        preprocessor.removeBody(laserCloudIn, laserCloudIn);
        // 포인트 클라우드 가공 및 생성
        getPointCloud();
        // 상황 출력
        report();
        // 포인트 클라우드 publish
        publishPointCloud();
        // 데이터 초기화
        clearMemory();
    }

   void getPointCloud() {
    start = std::chrono::high_resolution_clock::now();

    // 전체 포인트 클라우드 기본 셋팅
    pointCloudGenerator.getFullCloud(laserCloudIn, fullCloud, fullInfoCloud);
    
    // 관심 영역 설정 (후륜 기준으로 보정)
    pointCloudGenerator.getInterestCloud(fullCloud, interestCloud, {0.3, 4.3}, {-4, 4}, {0.3, 2});

    // 동적 장애물(큰 객체) 인식
    pointCloudGenerator.getObjectClusterCloud(interestCloud, smallObject_Cloud, bigObject_Cloud_vector);
    pointCloudGenerator.getObjectMarkers(bigObject_Cloud_vector, carBboxArray, carMarkerArray);

    // 콘(작은 객체) 인식 (후륜 기준으로 보정)
    pointCloudGenerator.getAngleCloud(smallObject_Cloud, coneClusterCloud, {0.3, 6.3}, {-3, 3}, {0.3, 2}, {-140, 140});

    pcl::PointCloud<PointType>::Ptr leftCone(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr rightCone(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr debugCone(new pcl::PointCloud<PointType>());
    clustering.identifyLRcone(coneClusterCloud, leftCone, rightCone, debugCone, midLineMarkerArray);

    // ── 긴급정지 debounce 로직 ──
    bool new_emergency = false;

    // 전방에 차량이 있는지 확인하는 ROI 범위 보정
    if (preprocessor.marker_is_in_area(carBboxArray, {1.3, 5.8}, {-1, 1}))
        new_emergency = true;

    // 시야 영역 포인트 확인 (후륜 기준으로 보정)
    pointCloudGenerator.getInterestCloud(smallObject_Cloud, fovCloud, {1.6, 3.1}, {-0.6, 0.6}, {0.3, 2});

    if (fovCloud->points.size() > 3)
        new_emergency = true;

    if (new_emergency != stable_emergency_state) {
        emergency_counter++;
        if (emergency_counter >= 4) {
            stable_emergency_state = new_emergency;
            emergency_counter = 0;
        }
    } else {
        emergency_counter = 0;
    }
    emergency.data = stable_emergency_state;
    // ────────────────────────────

    end = std::chrono::high_resolution_clock::now();
}

    void publishPointCloud() {
        // 좌표 및 시간 publish
        pubOdometry.publish(local);

        // MarkerArray Publish
        publisherMarkerArray(carMarkerArray, pubCarMarkerArray, "map");
        publisherMarkerArray(carBboxArray, pubCarBboxArray, "map");
        publisherMarkerArray(objMarkerArray, pubObjMarkerArray, "map");
        publisherMarkerArray(angleMarkerArray, pubAngleMarkerArray, "map");
        publisherMarkerArray(midLineMarkerArray, pubMidLineMarkerArray, "map"); //new

    
        // 포인트 클라우드 Publish
        publisher(fullCloud, pubFullCloud, "map");
        publisher(fullInfoCloud, pubFullInfoCloud, "map");
        publisher(interestCloud, pubInterestCloud, "map");
        publisher(interestCloud2D, pubInterestCloud2D, "map");
        publisher(ROICloud, pubROICloud, "map");
        publisher(fovCloud, pubFovCloud, "map");
        publisher(groundCloud, pubgroundCloud, "map");
        publisher(nongroundCloud, pubnongroundCloud, "map");
        publisher(debugCloud, pubDebugCloud, "map");
        publisher(transformedCloud, pubTransformedCloud, "map");
        publisher(transformedRoiCloud, pubTransformedRoiCloud, "map");
        publisher(coneClusterCloud, pubConeClusterCloud, "map");
        publisher(angleCloud, pubAngleCloud, "map");


        pubEmergencyFromLidar.publish(emergency);
    }
    // publisherMarkerArray()는 MarkerArray를 특정 퍼블리셔로 publish하는 헬퍼 함수입니다.
    void publisherMarkerArray(const std::shared_ptr<visualization_msgs::MarkerArray>& markerArray,
                                const ros::Publisher& pub,
                                const std::string& frame_id) {
        // 각 마커의 프레임과 타임스탬프 업데이트
        for (auto &marker : markerArray->markers) {
            marker.header.frame_id = frame_id;
            marker.header.stamp = ros::Time::now();
        }
        pub.publish(*markerArray);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "practics_dynamic_EV");
    ImageProjection imageProjection;
    ros::spin();
}
