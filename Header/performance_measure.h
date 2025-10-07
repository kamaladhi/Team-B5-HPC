#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>

class PerformanceMeasurement {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    
public:
    void startTimer();
    void stopTimer();
    double getElapsedMilliseconds();
    double getElapsedMicroseconds();
    void printResults(const std::string& operationName);
    
    static size_t getMemoryUsage();
    static double calculateMSE(const cv::Mat& img1, const cv::Mat& img2);
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    static bool validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels);
};

#endif