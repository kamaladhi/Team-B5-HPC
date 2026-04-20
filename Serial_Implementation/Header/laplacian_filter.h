#ifndef LAPLACIAN_FILTER_H
#define LAPLACIAN_FILTER_H

#include <opencv2/opencv.hpp>

class LaplacianFilterSerial {
public:
    cv::Mat apply(const cv::Mat& input);
};

class LaplacianFilterOMP {
public:
    cv::Mat apply(const cv::Mat& input);
};

#endif
