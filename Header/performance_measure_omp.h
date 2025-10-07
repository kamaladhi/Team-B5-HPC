
// performance_measure_omp.h (Add this to your header file)
#ifndef PERFORMANCE_MEASURE_OMP_H
#define PERFORMANCE_MEASURE_OMP_H

#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <map>
#include <vector>

struct VariantMetrics {
    std::string variantName;
    double totalTime;
    double avgTime;
    double minTime;
    double maxTime;
    int numExecutions;
    double cacheHitRate; // Optional
    double threadUtilization; // Optional
};

class PerformanceMeasurementOMP {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    double ompStartTime;
    double ompEndTime;
    
    // Variant tracking
    std::map<std::string, std::vector<double>> variantTimings;

public:
    void startTimer();
    void stopTimer();
    
    double getElapsedMilliseconds();
    double getElapsedMicroseconds();
    double getOMPElapsedSeconds();
    
    // Variant-specific methods
    void recordVariantTiming(const std::string& variantName, double time);
    VariantMetrics getVariantMetrics(const std::string& variantName);
    std::vector<VariantMetrics> getAllVariantMetrics();
    void printVariantComparison();
    void saveVariantReport(const std::string& filepath);
    
    // Existing methods
    void printResults(const std::string& operationName, int numThreads = -1);
    void printDetailedResults(const std::string& operationName, int numThreads, double serialTime);
    
    size_t getMemoryUsage();
    
    double calculateMSE(const cv::Mat& img1, const cv::Mat& img2);
    double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    
    bool validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels);
    
    void analyzeThreadUtilization();
    void printThreadInfo();
    
    double calculateSpeedup(double serialTime, double parallelTime);
    double calculateEfficiency(double speedup, int numThreads);
    
    // New comparison methods
    void compareVariants(const std::vector<std::string>& variantNames);
    double getSpeedupBetweenVariants(const std::string& baseVariant, const std::string& compareVariant);
};

#endif // PERFORMANCE_MEASURE_OMP_H