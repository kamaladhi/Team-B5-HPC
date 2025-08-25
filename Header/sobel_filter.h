#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <opencv2/opencv.hpp>

class SobelFilterSerial {
public:
    cv::Mat apply(const cv::Mat& input);
};

class SobelFilterOMP {
public:
    cv::Mat apply(const cv::Mat& input);
    cv::Mat applyOptimized(const cv::Mat& input); // <-- add this
};


#endif
