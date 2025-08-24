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
    std::chrono::high_resolution_clock::time_point startTime; // used for storing the start time.
    std::chrono::high_resolution_clock::time_point endTime; // used for storing the stop time.
    double ompStartTime;
    double ompEndTime;
    
public:
    void startTimer();
    void stopTimer();
    double getElapsedMilliseconds(); // used to return time lapsed in millisecond
    double getElapsedMicroseconds(); //
    double getOMPElapsedSeconds();
    void printResults(const std::string& operationName, int numThreads = 0); // used for printing the summary of the performance  name of operation and no of thread.
    void printDetailedResults(const std::string& operationName, int numThreads, double serialTime = 0.0); // this is along with serial time 
    
    static size_t getMemoryUsage();
    static double calculateMSE(const cv::Mat& img1, const cv::Mat& img2); // takes image 1 and image 2 and calculates the MSE
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);// takes image 1 and image 2 and calculates the PSNR
    static bool validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels); // Checks if a result matrix matches expected dimensions and channels.
    
    // Thread analysis functions
    static void analyzeThreadUtilization(); //Checks how well the threads are being utilized during a parallel operation.
    static void printThreadInfo();// prints information regarding the thread utilisation 
    static double calculateSpeedup(double serialTime, double parallelTime); // takes serial time and parallel time and calculates speed up by calculating ratio.
    static double calculateEfficiency(double speedup, int numThreads); //calculates the efficiency
};

#endif // PERFORMANCE_MEASURE_OMP_H
