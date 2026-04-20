#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <opencv2/opencv.hpp>

class GaussianFilterSerial {
public:
    cv::Mat apply(const cv::Mat& input, int kernelSize, double sigma);
};

class GaussianFilterOMP {
public:
    cv::Mat apply(const cv::Mat& input, int kernelSize, double sigma);
    cv::Mat applyMultiChannel(const cv::Mat& input, int kernelSize, double sigma); // <-- add this
};



#endif