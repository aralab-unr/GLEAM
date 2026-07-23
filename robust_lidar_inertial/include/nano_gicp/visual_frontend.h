#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct LidarFrame {
    cv::Mat intensity_image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors; // For matching later
};

class VisualFrontend {
public:
    VisualFrontend() {
        // Initialize ORB Feature Detector (Robust, fast, good for rotation)
        // You can also use cv::FastFeatureDetector for pure speed
        detector_ = cv::ORB::create(500); // Target 500 features
    }

    // Main function to process cloud
    LidarFrame processCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, 
                            int H, int W);

private:
    cv::Ptr<cv::FeatureDetector> detector_;

    // Hardware-Specific Projection (Fastest, requires 'ring' field or ordered cloud)
    cv::Mat projectHardware(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, int H, int W);
    
    // Fallback Math Projection (Slower, works on any cloud)
    cv::Mat projectSoftware(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, int H, int W);
};