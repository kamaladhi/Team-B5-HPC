#ifndef SHARPENING_FILTER_H
#define SHARPENING_FILTER_H

#include <opencv2/opencv.hpp>

class SharpeningFilterSerial {
public:
    cv::Mat apply(const cv::Mat& input);
};

class SharpeningFilterOMP {
public:
    cv::Mat apply(const cv::Mat& input);
};

#endif
