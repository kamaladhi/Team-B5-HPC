// performance_measure_omp.h
#ifndef PERFORMANCE_MEASURE_OMP_H
#define PERFORMANCE_MEASURE_OMP_H

#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>

class PerformanceMeasurementOMP {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    double ompStartTime;
    double ompEndTime;
    
public:
    void startTimer();
    void stopTimer();
    double getElapsedMilliseconds();
    double getElapsedMicroseconds();
    double getOMPElapsedSeconds();
    void printResults(const std::string& operationName, int numThreads = 0);
    void printDetailedResults(const std::string& operationName, int numThreads, double serialTime = 0.0);
    
    static size_t getMemoryUsage();
    static double calculateMSE(const cv::Mat& img1, const cv::Mat& img2);
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    static bool validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels);
    
    // Thread analysis functions
    static void analyzeThreadUtilization();
    static void printThreadInfo();
    static double calculateSpeedup(double serialTime, double parallelTime);
    static double calculateEfficiency(double speedup, int numThreads);
};

#endif // PERFORMANCE_MEASURE_OMP_H
