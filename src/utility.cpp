#include "utility.h"




/**
 ** 목적
    ** 1. 상수 및 구조체 저장
    ** 2. preprocessor, clustering ... 등 해당 카테고리에 해당되지 않는 함수
*/


// 상수 선언
    // ring 사용 여부
    const bool useCloudRing = true;                         // ring 사용 여부

    // 기호 상수
    const double PI = 3.14159265;                           // PI
    const int INF = 99999;                                  // 포인트의 범위를 발산으로 하고 싶을 때 넣을 것

    // 라이다 센서 상수
    const int N_SCAN = 32;                                  // 라이다의 row 이자 ring
    const int Horizon_SCAN = 1024;                          // 라이다의 colmun
    const float ang_res_x = 360.0/float(Horizon_SCAN);      // col -> 각도
    const float ang_res_y = 45/float(N_SCAN-1);             // row -> 각도
    const float ang_bottom = 20+0.1;                        // 라이다 시야각 / 2
    const int groundScanInd = 15;                           // 라이다 시야각 상하 분기각

    // 마운트 상수
    const double LI_TO_GPS_X = 0.65;                       // 라이다와 GPS 사이 거리
    const double LI_TO_GND_Z = -0.9;                        // 라이다와 지변 사이 거리
    const double LI_ROT_Y = 0;                              // y축 회전
    const double CAR_TO_GPS_X = -0.1;                       // 차량의 중심과 GPS사이 거리      
    
    // 차량 크기
    const float CAR_SIZE_X = 1.9;                          // 차량의 x축 길이
    const float CAR_SIZE_Y = 4.7;                          // 차량의 y축 길이
    const float CAR_SIZE_Z = 1.7;                          // 차량의 z축 길이

    // 계산 상수
    const double DEG_TO_RAD = PI/180.0;                     // 각도 -> 라이안
    const double RAD_TO_DEG = 180.0/PI;                     // 라이안 -> 각도

    // 콘 클러스터링 상수
    const double Cone_Tolerance = 0.2;                      // 콘 사이 거리
    const int CONE_min_SIZE = 2;                           // 콘에 찍히는 포인트 최소
    const int CONE_MAX_SIZE = 3000;                          // 콘에 찍히는 포인트 최대

    // 기본 클러스터링 상수
    const double OBJECT_Tolerance = 0.2;                      // 객체 사이 거리
    const int OBJECT_min_SIZE = 2;                           // 객체에 찍히는 포인트 최소
    const int OBJECT_MAX_SIZE = 3000;                          // 객체에 찍히는 포인트 최대

    // 콘 주행 상수
    const double CONE_BETWEEN = 2;                        // 차량과 양측에 위치한 콘 사이 y방향 거리

// 서브 프로세스
    pid_t subProcess_obj(int camera_idx, G_CMD g_cmd , std::vector<string> camera_pair)
    {
		int pipefd[2];
		if (pipe(pipefd) == -1) {
			perror("pipe");
			exit(EXIT_FAILURE);
		}

		pid_t pid = fork();
		if (pid == -1) {
			perror("fork");
			exit(EXIT_FAILURE);
		}

		if (pid == 0) {
			// 자식 프로세스
			close(pipefd[1]);
			dup2(pipefd[0], STDIN_FILENO);

			// 실행 파일 존재 확인
            if (access("/home/sadgod33/catkin_ws/devel/lib/pcl_cpp_tutorial/fusion_object", X_OK) == -1) {
                perror("access");
                exit(EXIT_FAILURE);
            }

            // 실행 파일 실행
            execl("/home/sadgod33/catkin_ws/devel/lib/pcl_cpp_tutorial/fusion_object", "fusion_object", (char *)NULL);
			
			// execl 실행 실패 시
			perror("execl");
			exit(EXIT_FAILURE);
		} else {
			// 부모 프로세스
			close(pipefd[0]);

			// 인덱스 송신
			write(pipefd[1], &camera_idx, sizeof(camera_idx));

			// rvec
			write(pipefd[1], g_cmd.rvec.data(), 3 * sizeof(double));
			// t_mat
			write(pipefd[1], g_cmd.t_mat.data(), 3 * sizeof(double));
			// focal_length
			write(pipefd[1], g_cmd.focal_length.data(), 2 * sizeof(double));
			// cam
			write(pipefd[1], g_cmd.cam.data(), 2 * sizeof(int));

            // string 송신
            uint32_t image_topic_size = camera_pair[0].size();
            uint32_t box_topic_size = camera_pair[1].size();
            uint32_t marker_topic_size = camera_pair[2].size();
            write(pipefd[1], &image_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[0].c_str(), image_topic_size);
            write(pipefd[1], &box_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[1].c_str(), box_topic_size);
            write(pipefd[1], &marker_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[2].c_str(), marker_topic_size);

			
			close(pipefd[1]);
			return pid;
		}
		return -1;
    }

    pid_t subProcess_line(int camera_idx, G_CMD g_cmd , std::vector<string> camera_pair)
    {
		int pipefd[2];
		if (pipe(pipefd) == -1) {
			perror("pipe");
			exit(EXIT_FAILURE);
		}

		pid_t pid = fork();
		if (pid == -1) {
			perror("fork");
			exit(EXIT_FAILURE);
		}

		if (pid == 0) {
			// 자식 프로세스
			close(pipefd[1]);
			dup2(pipefd[0], STDIN_FILENO);

			// 실행 파일 존재 확인
            if (access("/home/sadgod33/catkin_ws/devel/lib/pcl_cpp_tutorial/fusion_line", X_OK) == -1) {
                perror("access");
                exit(EXIT_FAILURE);
            }

            // 실행 파일 실행
            execl("/home/sadgod33/catkin_ws/devel/lib/pcl_cpp_tutorial/fusion_line", "fusion_line", (char *)NULL);
        
			// execl 실행 실패 시
			perror("execl");
			exit(EXIT_FAILURE);
		} else {
			// 부모 프로세스
			close(pipefd[0]);

			// 인덱스 송신
			write(pipefd[1], &camera_idx, sizeof(camera_idx));

			// rvec
			write(pipefd[1], g_cmd.rvec.data(), 3 * sizeof(double));
			// t_mat
			write(pipefd[1], g_cmd.t_mat.data(), 3 * sizeof(double));
			// focal_length
			write(pipefd[1], g_cmd.focal_length.data(), 2 * sizeof(double));
			// cam
			write(pipefd[1], g_cmd.cam.data(), 2 * sizeof(int));

            // string 송신
            uint32_t image_topic_size = camera_pair[0].size();
            uint32_t seg_image_topic_size = camera_pair[1].size();
            uint32_t line_topic_size = camera_pair[2].size();
            uint32_t follow_topic_size = camera_pair[3].size();

            write(pipefd[1], &image_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[0].c_str(), image_topic_size);
            write(pipefd[1], &seg_image_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[1].c_str(), seg_image_topic_size);
            write(pipefd[1], &line_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[2].c_str(), line_topic_size);
            write(pipefd[1], &follow_topic_size, sizeof(uint32_t));
            write(pipefd[1], camera_pair[3].c_str(), follow_topic_size);

			close(pipefd[1]);
			return pid;
		}
		return -1;
    }

// 객체 저장 포인터 클라우드 선언
    pcl::PointCloud<PointType>::Ptr clusterStopOver(new pcl::PointCloud<PointType>());


// 함수 선언
    // 현 시각 저장
    ros::Time rosTime;
    void saveCurrentTime(const sensor_msgs::PointCloud2::ConstPtr& rosCloud)
    {
        rosTime = rosCloud->header.stamp;
    }

    // map을 읽고 global_path에 저장
    std::vector<std::pair<float, float>> global_path = {};
    int curr_index = 0;
    void mapReader(const std::string& map_json) 
    {
        Json::Value root;		
        Json::Reader reader;
        std::ifstream t;
        std::string index;
        t.open(map_json);
        if (!reader.parse(t, root)) {
            cout << "Parsing Failed" << endl;
        }
        for (int k=0;k<root.size();++k){
            index = to_string(k);
            double x = root[index][0].asDouble();
            double y = root[index][1].asDouble();
            global_path.emplace_back(x, y);
        }
        curr_index = 0;
    }

    // global_path내 위치한 curr_index를 구함
    void findCurrIndex()
    {
        double min_dis = -1;
        int min_idx = 0;
        int step_size = 500;
        for (int i = std::max(curr_index-step_size, min_idx); i < (curr_index+step_size); ++i) 
        {
            double dis = std::hypot(global_path[i].first - ego_x, global_path[i].second - ego_y);
            if (min_dis > dis || min_dis == -1)
            {
                min_dis = dis;
                min_idx = i;
            }
        }
        curr_index = min_idx;
    }

    // 현 자동차의 GPS 및 해딩을 전역 상수로 저장
    double ego_x = 0;
    double ego_y = 0;
    double ego_heading_deg = 0;  // 각도
    double ego_heading_rad = 0;  // 라디안
    void setEgo(const geometry_msgs::PoseStampedConstPtr& poseMsg, geometry_msgs::PoseStamped& local)
    {
        ego_x = poseMsg->pose.position.x;
        ego_y = poseMsg->pose.position.y;
        ego_heading_deg = poseMsg->pose.orientation.w;
        ego_heading_rad = poseMsg->pose.orientation.w * DEG_TO_RAD;

        local.header.stamp = rosTime;
        local.header.frame_id = "base_laser";
        local.pose.position.x = poseMsg->pose.position.x;
        local.pose.position.y = poseMsg->pose.position.y;
        local.pose.orientation.w = poseMsg->pose.orientation.w;
    }

    // 터널에서의 해딩값을 전역 상수로 저장
    std::tuple<double, int> setHeadingInTunnel(const pcl::PointCloud<PointType>::Ptr& input_pointCloud, 
                              const std::pair<double, double>        x_threshold, 
                              const std::pair<double, double>        y_threshold, 
                              const std::pair<double, double>        z_threshold)
    {
        pcl::PointCloud<PointType>::Ptr temp_pointCloud(new pcl::PointCloud<PointType>);
        temp_pointCloud->clear();
        for(PointType& point : input_pointCloud->points){
            if(x_threshold.first < point.x && point.x < x_threshold.second &&
                y_threshold.first < point.y && point.y < y_threshold.second &&
                z_threshold.first < point.z && point.z < z_threshold.second)
            {
                temp_pointCloud->points.push_back(point);
            }
        }

    // 다운샘플링을 진행합니다.
        pcl::VoxelGrid<PointType> voxelGrid;
        voxelGrid.setInputCloud(temp_pointCloud);
        voxelGrid.setLeafSize(0.1, 0.1, 0.1);
        voxelGrid.filter(*temp_pointCloud);

    // 8개의 벡터를 생성하여 각 영역의 포인트들을 저장합니다.
        std::unique_ptr<std::vector<pcl::PointCloud<PointType>>> regions (new std::vector<pcl::PointCloud<PointType>>(8));
        for (PointType& point : temp_pointCloud->points)
        {
            int regionIndex = 0;
            if (point.y < -1) {
                // x 값에 따른 인덱스 결정 (0-2.5, 2.5-5, 5-7.5, 7.5-10)
                if (point.x >= 0 && point.x < 2.5) {
                    regionIndex = 0;
                }
                if (point.x >= 1.25 && point.x < 3.75) {
                    regionIndex = 1;
                }
                if (point.x >= 2.5 && point.x < 5) {
                    regionIndex = 2;
                }
                if (point.x >= 3.75 && point.x < 6.25) {
                    regionIndex = 3;
                }
                if (point.x >= 5 && point.x < 7.5) {
                    regionIndex = 4;
                }
                if (point.x >= 6.25 && point.x < 8.75) {
                    regionIndex = 5;
                }
                if (point.x >= 7.5 && point.x <= 10) {
                    regionIndex = 6;
                }
                // 해당 영역에 포인트 추가
                regions->at(regionIndex).push_back(point);
            }
        }

    // 각 영역별 선형회귀를 진행합니다.
        std::shared_ptr<std::vector<float>> x_values = std::make_shared<std::vector<float>>();
        std::shared_ptr<std::vector<float>> y_values = std::make_shared<std::vector<float>>();
        std::shared_ptr<std::vector<double>> coeffs_values = std::make_shared<std::vector<double>>();
        std::shared_ptr<std::vector<double>> coeffs_values_y = std::make_shared<std::vector<double>>();
        
        for (size_t i = 0; i < regions->size(); ++i)
        {
            x_values->clear();
            y_values->clear();
            pcl::PointCloud<PointType>& region_cloud = (*regions)[i];
            for (PointType& point : region_cloud)
            {
                x_values->push_back(point.x);
                y_values->push_back(point.y);
            }

            if (x_values->size() <= 2)
            {
                continue;
            }

            const int n = x_values->size();
            Eigen::MatrixXd X(n, 2);
            Eigen::VectorXd Y(n);

            for (int i = 0; i < n; ++i)
            {
                X(i, 0) = (*x_values)[i];
                X(i, 1) = 1;
                Y(i)    = (*y_values)[i];
            }
            
            Eigen::VectorXd coeffs = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
            coeffs_values->push_back(coeffs[0]);
            coeffs_values_y->push_back(coeffs[1]);
        }

    // 터널인지 판단 및 진입 혹은 퇴장 판단
        std::shared_ptr<std::vector<double>> region_dis = std::make_shared<std::vector<double>>();
        double sum_distance = 0.0;
        for (size_t i = 0; i < regions->size(); ++i)
        {
            pcl::PointCloud<PointType>& region_cloud = (*regions)[i];
            for (PointType& point : region_cloud)
            {
                // 완벽히 거리를 구하기 보다는 대략적으로 구하는게 가볍기도 하고, 터널 안에서 기울기가 +-5도 내외라 괜찮을거라고 판단
                sum_distance += std::abs(point.y - ((*coeffs_values)[i]*point.x + (*coeffs_values_y)[i]));
            }
            sum_distance /= region_cloud.size();
            region_dis->push_back(sum_distance);
        }

        unsigned int tunnelCount = 0;
        unsigned int tunnelLoc = 0;
        for (size_t i = 0; i < region_dis->size(); ++i)
        {
            if (region_dis->at(i) <= 0.7)
            {
                tunnelCount++;
                tunnelLoc += pow(10, i);
            }
        }
        if (tunnelCount < 5)
        {
            return {-1, tunnelLoc};
        }

    // 총합 기울기 구하기
        double coeffs_sum = 0;
        if (coeffs_values->size() >= 3)
        {
            std::sort(coeffs_values->begin(), coeffs_values->end());
            for (int i = 1; i < coeffs_values->size()-1; ++i)  // 8개 중 최소 1개, 최대 1개를 제외하고 중간 6개를 합산
            { 
                coeffs_sum += atan((*coeffs_values)[i]);
            }
            coeffs_sum /= (coeffs_values->size()-2);
            return {coeffs_sum, tunnelLoc};
        }
        else
        {
            for (int i = 0; i < coeffs_values->size(); ++i)
            { 
                coeffs_sum += atan((*coeffs_values)[i]);
            }
            coeffs_sum /= coeffs_values->size();
            return {coeffs_sum, tunnelLoc};
        }
    }

    // 한 포인트의 좌표를 상대에서 절대로 변환
    void transformRelToAbs(PointType& point)
    {
        double tempX = cos(ego_heading_rad) * point.x - sin(ego_heading_rad) * point.y + ego_x;
        double tempY = sin(ego_heading_rad) * point.x + cos(ego_heading_rad) * point.y + ego_y;
        
        point.x = tempX;
        point.y = tempY;
    }


    // 한 포인트의 좌표를 절대에서 상대로 변환
    void transformAbsToRel(PointType& point)
    {
        double tempX = cos(ego_heading_rad) * (point.x - ego_x) + sin(ego_heading_rad) * (point.y - ego_y);
        double tempY = -sin(ego_heading_rad) * (point.x - ego_x) + cos(ego_heading_rad) * (point.y - ego_y);
        
        point.x = tempX;
        point.y = tempY;
    }

    // markerArray 초기화
    void deleteAllMarkers(const std::shared_ptr<visualization_msgs::MarkerArray>& markerArray) {
        // 모든 마커의 액션을 DELETE로 설정
        for (size_t i = 0; i < markerArray->markers.size(); ++i) {
            markerArray->markers[i].action = visualization_msgs::Marker::DELETE;
        }
        markerArray->markers.clear();
    }


    // publish 실행
    void publisher(const pcl::PointCloud<PointType>::Ptr& pclCloud, const ros::Publisher& publisher, const std::string frame_id)
    {
        std::unique_ptr<sensor_msgs::PointCloud2> tempCloud(new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*pclCloud, *tempCloud);
        tempCloud->header.frame_id = frame_id;
        tempCloud->header.stamp = rosTime;
        publisher.publish(*tempCloud);
	}


    // markerarray용 publish 실행
    void publisherMarkerArray(const std::shared_ptr<visualization_msgs::MarkerArray>& markerArray, const ros::Publisher& publisher, const std::string frame_id)
    {
        for (auto& marker : markerArray->markers) 
        {
            marker.header.frame_id = frame_id;
            marker.header.stamp = rosTime;
        }
        publisher.publish(*markerArray);
	}


// 칼만필터 클래스 선언
    KalmanFilter::KalmanFilter() 
    {
        identity = Eigen::Matrix2f::Identity();
        errorCovariance = Eigen::Matrix2f::Identity();
        measurementNoise = Eigen::Matrix2f::Identity() * 0.1;
        processNoise = Eigen::Matrix2f::Identity() * 0.1;
    }

    void KalmanFilter::init(Eigen::Vector2f initialState) 
    {
        state = initialState;
    }

    void KalmanFilter::predict() 
    {
        errorCovariance = errorCovariance + processNoise;
    }

    void KalmanFilter::update(Eigen::Vector2f measurement) 
    {
        Eigen::Matrix2f kalmanGain = errorCovariance * (errorCovariance + measurementNoise).inverse();
        state = state + kalmanGain * (measurement - state);
        errorCovariance = (identity - kalmanGain) * errorCovariance;
    } 


// 오브젝트 인식 코드
    std::tuple<double, double, double, double> euler_to_quaternion(const double &theta)
    {
        double yaw = M_PI/2 + theta;
        double cr = cos(0);
        double sr = sin(0);
        double cp = cos(0);
        double sp = sin(0);
        double cy = cos(yaw*0.5);
        double sy = sin(yaw*0.5);
        
        double ori_w = cr * cp * cy + sr * sp * sy;
        double ori_x = sr * cp * cy - cr * sp * sy;
        double ori_y = cr * sp * cy + sr * cp * sy;
        double ori_z = cr * cp * sy - sr * sp * cy;    
        return std::make_tuple(ori_x, ori_y, ori_z, ori_w);
    }


    double endpoint_distance(const PointType &point, const std::pair<double, double> minpoint, const std::pair<double, double> maxpoint)
    {
        //Point 에는 cluster안에 들어있는 모든 점의 정보를 담고 있어야 하고
        double dx = maxpoint.first - minpoint.first;
        double dy = maxpoint.second - minpoint.second;
        double diag_dist = sqrt(dx*dx + dy*dy);
        // y = ax + c 여기서 점과 직선 사이의 거리를 반환하면
        double a = (maxpoint.second - minpoint.second)/(maxpoint.first - minpoint.first);
        double c = maxpoint.second - a*maxpoint.first;
        double dist = abs(a*point.x - point.y + c)/sqrt(a*a + 1); //여기 point 받아오는 부분 수정
        return dist;
        }


    std::tuple<double, double> find_position(const pcl::PointCloud<PointType> &cloud, double threshold, double min_y, double max_y, double theta)
    {
        double dist_max = 0.0;
        size_t max_dist_ind = 0;
        size_t end = cloud.points.size();
        
        std::pair<double, double> minpoint;
        std::pair<double, double> maxpoint;
        double position_x, position_y;
        for (size_t i=0; i<end; i++){
            if (cloud.points[i].y == min_y){
                minpoint.first = cloud.points[i].x; 
                minpoint.second = cloud.points[i].y;
            }
            if (cloud.points[i].y == max_y){
                maxpoint.first = cloud.points[i].x;
                maxpoint.second = cloud.points[i].y;
            }
        }

        for (size_t i = 0; i<end; i++){
            double dist = endpoint_distance(cloud.points[i], minpoint, maxpoint);
            if (dist > dist_max){
                max_dist_ind =i;
                dist_max = dist;
            } 
        }

        if (dist_max > threshold){ // 두꺼운 객체 => 돌출점으로
            // slop = (minpoint.first - cloud.points[max_dist_ind].x)/(minpoint.second-cloud.points[max_dist_ind].y);
            position_x = cloud.points[max_dist_ind].x, position_y = cloud.points[max_dist_ind].y;
            
        }
        else // 기둥처럼 얇은 객체 => 중심 점으로
        {
            // slop = FLT_MAX;
            position_x = (maxpoint.first + minpoint.first)/2, position_y = (maxpoint.second + minpoint.second)/2;
        }

        return std::make_tuple(position_x, position_y); // 나중에 slop를 orientation으로 바꾸어 주는 코드 추가 필요
    }


    bool compare(std::pair<float, float> a, std::pair<float, float> b)
    {
        return a.first < b.first;
    }


    double variance_criterion(const cv::Mat& c1, const cv::Mat& c2)
    {

        std::vector<double> c1_deep;
        std::vector<double> c2_deep;

        for (size_t i = 0; i<c1.rows; ++i){
            for (size_t j =0; j<c1.cols; ++j){
                c1_deep.push_back(c1.at<double>(i,j));
                c2_deep.push_back(c2.at<double>(i,j));
            }
        }
        sort(c1_deep.begin(), c1_deep.end());
        sort(c2_deep.begin(), c2_deep.end());

        int n_c1 = c1_deep.size();
        int n_c2 = c2_deep.size();

        double c1_min = c1_deep[0];
        double c2_min = c2_deep[0];

        double c1_max = c1_deep[n_c1 - 1];
        double c2_max = c2_deep[n_c2 - 1];

        std::vector<double> D1;
        for (double ic1 : c1_deep){
            double dist1 = std::min(sqrt(pow((c1_max - ic1), 2)), sqrt(pow((ic1 - c1_min), 2)));
            D1.push_back(dist1);
        }

        std::vector<double> D2;
        for (double ic2 : c2_deep){
            double dist2 = std::min(sqrt(pow((c2_max - ic2), 2)), sqrt(pow((ic2 - c2_min), 2)));
            D2.push_back(dist2);
        }

        std::vector<double> E1, E2;
        for (size_t i=0; i<D1.size();++i){
            double d1 = D1[i];
            double d2 = D2[i];
            if (d1<d2){
                E1.push_back(d1);
            }
            else{
                E2.push_back(d2);
            }
        }
        //분산 계산하기
        double V1 = 0.0;
        if (!E1.empty()){
            double mean1 = 0.0;
            for (double e1 : E1){
                mean1 += e1;
            }
            mean1 /=E1.size();

            for (double e1 : E1){
                V1+= std::pow(e1-mean1,2);
            }
            V1 = -V1/E1.size();
        }

        double V2 = 0.0;
        if (!E2.empty()){
            double mean2 = 0.0;
            for (double e2 : E2){
                mean2 += e2;
            }
            mean2 /=E2.size();

            for (double e2 : E2){
                V2+= std::pow(e2-mean2,2);
            }
            V2 = -V2/E2.size();
        }
        //더해서 cost 계산
        double gamma = V1+V2;
        return gamma;
    }


    std::tuple<double, double, double, double> get_best_theta(const pcl::PointCloud<PointType> &cloud)
    {

        cv::Mat Matrix_pts = cv::Mat::zeros(cloud.points.size(), 2, CV_64FC1);
        double mean_y = 0;
        double mean_x = 0;
        for (size_t  i=0;i<cloud.size();++i){
            Matrix_pts.at<double>(i,0) = cloud.points[i].x;
            Matrix_pts.at<double>(i,1) = cloud.points[i].y;
            if (cloud.points[i].z < 1.3){
                mean_y += cloud.points[i].y;
                mean_x += cloud.points[i].x;
            }

        }
        mean_x = mean_x / cloud.size();
        mean_y = mean_y / cloud.size();
        // std::cout << mean_x << " : " << mean_y << std::endl;
        double dtheta = M_PI/180;
        double minimal_cost = (-1.0)*FLT_MAX;
        double best_theta = FLT_MAX;
        double second_theta = FLT_MAX;
        double ori_x, ori_y, ori_z, ori_w;
        cv::Mat e1 = cv::Mat::zeros(1,2,CV_64FC1);
        cv::Mat e2 = cv::Mat::zeros(1,2,CV_64FC1);

        for (size_t k=0;k<89;++k){
            double theta = k*M_PI/180;
            double cost = (-1.0)*FLT_MAX;
            if (theta < 89*M_PI/180){
                e1.at<double>(0,0) = cos(theta);
                e1.at<double>(0,1) = sin(theta);
                e2.at<double>(0,0) = -sin(theta);
                e2.at<double>(0,1) = cos(theta);

                cv::Mat c1 = Matrix_pts*e1.t();
                cv::Mat c2 = Matrix_pts*e2.t();

                cost = variance_criterion(c1, c2);
                if (minimal_cost < cost){
                    minimal_cost = cost;
                    second_theta = best_theta;
                    best_theta = theta;
                }
            }
        }
        if (best_theta == FLT_MAX){
            best_theta = -M_PI/2;}
        

        return std::make_tuple(best_theta, mean_x, mean_y, best_theta);
    }


    // visualization_msgs::Marker bbox_3d(const pcl::PointCloud<PointType> &cloud)
    visualization_msgs::Marker bbox_3d(const pcl::PointCloud<PointType> &cloud, const double &heading, const double &enu_x,const double &enu_y, const int &Id)
    {
        pcl::PointXYZI minPoint, maxPoint;
        pcl::getMinMax3D(cloud, minPoint, maxPoint);
        visualization_msgs::Marker marker;
        // if ((maxPoint.x - minPoint.x > 0.2) && (maxPoint.y - minPoint.y < 3.0) ){


        double center_x = (maxPoint.x + minPoint.x)/2;
        double center_y = (maxPoint.y + minPoint.y)/2;
        double center_z = (maxPoint.z + minPoint.z)/2;

        double length_x = fabs(maxPoint.x - minPoint.x);
        double length_y = fabs(maxPoint.y - minPoint.y);
        double length_z = fabs(maxPoint.z - minPoint.z);

        marker.id = 0;
        
        marker.type = visualization_msgs::Marker::LINE_LIST;
        
        marker.action = visualization_msgs::Marker::ADD;

        // 8개의 pos에 대해서 모두 좌표를 convert해서 geometry에 담자.
        geometry_msgs::Point p[8];
        p[0].x = center_x - length_x / 2; p[0].y = center_y - length_y / 2; p[0].z = center_z - length_z / 2;
        p[1].x = center_x + length_x / 2; p[1].y = center_y - length_y / 2; p[1].z = center_z - length_z / 2;
        p[2].x = center_x - length_x / 2; p[2].y = center_y + length_y / 2; p[2].z = center_z - length_z / 2;
        p[3].x = center_x + length_x / 2; p[3].y = center_y + length_y / 2; p[3].z = center_z - length_z / 2;
        p[4].x = center_x - length_x / 2; p[4].y = center_y - length_y / 2; p[4].z = center_z + length_z / 2;
        p[5].x = center_x + length_x / 2; p[5].y = center_y - length_y / 2; p[5].z = center_z + length_z / 2;
        p[6].x = center_x - length_x / 2; p[6].y = center_y + length_y / 2; p[6].z = center_z + length_z / 2;
        p[7].x = center_x + length_x / 2; p[7].y = center_y + length_y / 2; p[7].z = center_z + length_z / 2;
        

        for(int i=0; i<8; i++) {
            double old_x = p[i].x;
            double old_y = p[i].y;

            p[i].x = old_x*cos(-ego_heading_rad) + old_y*sin(-ego_heading_rad) + enu_x;
            p[i].y = -old_x*sin(-ego_heading_rad) + old_y*cos(-ego_heading_rad) + enu_y;
        }
        int connections[12][2]={{0,1},{1,3},{3,2},{2,0}, {4,5},{5,7},{7,6},{6,4}, {0,4},{1,5},{3,7},{2,6}};

        for(int i=0;i<12;i++){
            // For each line connection defined above,
            // add both vertices to 'points' array in Marker message.
            marker.points.push_back(p[connections[i][0]]);
            marker.points.push_back(p[connections[i][1]]);
        }
        marker.id = Id;
        marker.lifetime = ros::Duration(0.1);
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker.scale.x = 0.05;

        return marker;
    }
