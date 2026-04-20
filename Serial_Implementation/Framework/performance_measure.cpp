#include "performance_measure.h"
#include <iostream>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

void PerformanceMeasurement::startTimer() {
    startTime = std::chrono::high_resolution_clock::now();
}

void PerformanceMeasurement::stopTimer() {
    endTime = std::chrono::high_resolution_clock::now();
}

double PerformanceMeasurement::getElapsedMilliseconds() {
    auto duration = endTime - startTime;
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

double PerformanceMeasurement::getElapsedMicroseconds() {
    auto duration = endTime - startTime;
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

void PerformanceMeasurement::printResults(const std::string& operationName) {
    std::cout << "=== PERFORMANCE RESULTS ===" << std::endl;
    std::cout << "Operation: " << operationName << std::endl;
    std::cout << "Execution time: " << getElapsedMilliseconds() << " ms" << std::endl;
    std::cout << "Execution time: " << getElapsedMicroseconds() << " μs" << std::endl;
    std::cout << "Memory usage: " << getMemoryUsage() << " KB" << std::endl;
}

size_t PerformanceMeasurement::getMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize / 1024;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
#endif
}

double PerformanceMeasurement::calculateMSE(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_64F);
    diff = diff.mul(diff);
    cv::Scalar mse = cv::sum(diff);
    double total_mse = mse[0] + mse[1] + mse[2];
    return total_mse / (img1.rows * img1.cols * img1.channels());
}

double PerformanceMeasurement::calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = calculateMSE(img1, img2);
    if (mse < 1e-10) {
        return 100.0;
    }
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}

bool PerformanceMeasurement::validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels) {
    if (result.empty()) {
        std::cout << "Validation failed: Result is empty" << std::endl;
        return false;
    }
    
    if (result.cols != expectedWidth || result.rows != expectedHeight) {
        std::cout << "Validation failed: Size mismatch. Expected: " << expectedWidth 
                  << "x" << expectedHeight << ", Got: " << result.cols << "x" << result.rows << std::endl;
        return false;
    }
    
    if (result.channels() != expectedChannels) {
        std::cout << "Validation failed: Channel mismatch. Expected: " << expectedChannels 
                  << ", Got: " << result.channels() << std::endl;
        return false;
    }
    
    std::cout << "Validation passed: " << result.cols << "x" << result.rows 
              << " with " << result.channels() << " channels" << std::endl;
    return true;
}
