
// performance_measure_omp.cpp
#include "performance_measure_omp.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

void PerformanceMeasurementOMP::startTimer() {
    startTime = std::chrono::high_resolution_clock::now();
    ompStartTime = omp_get_wtime();
}

void PerformanceMeasurementOMP::stopTimer() {
    endTime = std::chrono::high_resolution_clock::now();
    ompEndTime = omp_get_wtime();
}

double PerformanceMeasurementOMP::getElapsedMilliseconds() {
    auto duration = endTime - startTime;
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

double PerformanceMeasurementOMP::getElapsedMicroseconds() {
    auto duration = endTime - startTime;
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

double PerformanceMeasurementOMP::getOMPElapsedSeconds() {
    return ompEndTime - ompStartTime;
}

void PerformanceMeasurementOMP::printResults(const std::string& operationName, int numThreads) {
    std::cout << "=== PERFORMANCE RESULTS (OpenMP) ===" << std::endl;
    std::cout << "Operation: " << operationName << std::endl;
    if (numThreads > 0) {
        std::cout << "Threads used: " << numThreads << std::endl;
    }
    std::cout << "Execution time (chrono): " << getElapsedMilliseconds() << " ms" << std::endl;
    std::cout << "Execution time (OMP): " << std::fixed << std::setprecision(3) 
              << getOMPElapsedSeconds() * 1000 << " ms" << std::endl;
    std::cout << "Memory usage: " << getMemoryUsage() << " KB" << std::endl;
    std::cout << "CPU threads available: " << omp_get_max_threads() << std::endl;
}

void PerformanceMeasurementOMP::printDetailedResults(const std::string& operationName, int numThreads, double serialTime) {
    double parallelTime = getElapsedMilliseconds();
    
    std::cout << "=== DETAILED PERFORMANCE ANALYSIS ===" << std::endl;
    std::cout << "Operation: " << operationName << std::endl;
    std::cout << "Threads used: " << numThreads << std::endl;
    std::cout << "Parallel execution time: " << std::fixed << std::setprecision(2) 
              << parallelTime << " ms" << std::endl;
    
    if (serialTime > 0.0) {
        double speedup = calculateSpeedup(serialTime, parallelTime);
        double efficiency = calculateEfficiency(speedup, numThreads);
        
        std::cout << "Serial execution time: " << serialTime << " ms" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "Efficiency: " << std::fixed << std::setprecision(1) 
                  << efficiency * 100 << "%" << std::endl;
        
        // Theoretical maximum speedup (Amdahl's Law approximation)
        std::cout << "Theoretical max speedup: " << numThreads << "x" << std::endl;
        std::cout << "Speedup efficiency: " << std::fixed << std::setprecision(1)
                  << (speedup / numThreads) * 100 << "%" << std::endl;
    }
    
    std::cout << "Memory usage: " << getMemoryUsage() << " KB" << std::endl;
}

size_t PerformanceMeasurementOMP::getMemoryUsage() {
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

double PerformanceMeasurementOMP::calculateMSE(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff); //calculates absolute difference between the images 
    diff.convertTo(diff, CV_64F);// 64 bit floating point
    diff = diff.mul(diff); // squares the difference 
    cv::Scalar mse = cv::sum(diff); // sums the squared difference 
    double total_mse = mse[0] + mse[1] + mse[2]; // adds the sum of mse of all the channel
    return total_mse / (img1.rows * img1.cols * img1.channels());
}

double PerformanceMeasurementOMP::calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) { //Peak Signal-to-Noise Ratio
    double mse = calculateMSE(img1, img2);
    if (mse < 1e-10) {
        return 100.0;
    }
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}

bool PerformanceMeasurementOMP::validateResult(const cv::Mat& result, int expectedWidth, int expectedHeight, int expectedChannels) {
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
              << " with " << result.channels() << " channels" << std::endl; // returns number of channel
    return true;
}

void PerformanceMeasurementOMP::analyzeThreadUtilization() {
    std::cout << "\n=== THREAD UTILIZATION ANALYSIS ===" << std::endl;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " of " << num_threads << " is active" << std::endl; // tells which thread is active.
        }
    }
    
    std::cout << "Total threads used: " << omp_get_max_threads() << std::endl;
    std::cout << "Nested parallelism: " << (omp_get_nested() ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Dynamic thread adjustment: " << (omp_get_dynamic() ? "Enabled" : "Disabled") << std::endl;
}

void PerformanceMeasurementOMP::printThreadInfo() {
    std::cout << "\n=== OPENMP THREAD INFORMATION ===" << std::endl;
    std::cout << "OpenMP Version: " << _OPENMP << std::endl;
    std::cout << "Maximum threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
    
    #pragma omp parallel // enables multi thread
    {
        #pragma omp single // only one thread
        {
            std::cout << "Threads in current parallel region: " << omp_get_num_threads() << std::endl; // this will return the total number of threads in the current parallel region.
        }
    }
}

double PerformanceMeasurementOMP::calculateSpeedup(double serialTime, double parallelTime) {
    if (parallelTime <= 0.0) return 0.0;
    return serialTime / parallelTime;
}

double PerformanceMeasurementOMP::calculateEfficiency(double speedup, int numThreads) {
    if (numThreads <= 0) return 0.0;
    return speedup / numThreads;
}
