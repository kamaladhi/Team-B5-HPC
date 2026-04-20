// performance_measure_omp.cpp (Enhanced implementation)
#include "performance_measure_omp.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

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

// Variant-specific implementations
void PerformanceMeasurementOMP::recordVariantTiming(const std::string& variantName, double time) {
    variantTimings[variantName].push_back(time);
}

VariantMetrics PerformanceMeasurementOMP::getVariantMetrics(const std::string& variantName) {
    VariantMetrics metrics;
    metrics.variantName = variantName;
    
    if (variantTimings.find(variantName) == variantTimings.end() || variantTimings[variantName].empty()) {
        metrics.totalTime = 0.0;
        metrics.avgTime = 0.0;
        metrics.minTime = 0.0;
        metrics.maxTime = 0.0;
        metrics.numExecutions = 0;
        return metrics;
    }
    
    const std::vector<double>& times = variantTimings[variantName];
    
    metrics.numExecutions = times.size();
    metrics.totalTime = 0.0;
    metrics.minTime = times[0];
    metrics.maxTime = times[0];
    
    for (double time : times) {
        metrics.totalTime += time;
        metrics.minTime = std::min(metrics.minTime, time);
        metrics.maxTime = std::max(metrics.maxTime, time);
    }
    
    metrics.avgTime = metrics.totalTime / metrics.numExecutions;
    
    return metrics;
}

std::vector<VariantMetrics> PerformanceMeasurementOMP::getAllVariantMetrics() {
    std::vector<VariantMetrics> allMetrics;
    
    for (const auto& pair : variantTimings) {
        allMetrics.push_back(getVariantMetrics(pair.first));
    }
    
    // Sort by average time
    std::sort(allMetrics.begin(), allMetrics.end(), 
              [](const VariantMetrics& a, const VariantMetrics& b) {
                  return a.avgTime < b.avgTime;
              });
    
    return allMetrics;
}

void PerformanceMeasurementOMP::printVariantComparison() {
    std::cout << "\n=== CONVOLUTION VARIANT PERFORMANCE COMPARISON ===" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto allMetrics = getAllVariantMetrics();
    
    if (allMetrics.empty()) {
        std::cout << "No variant data recorded." << std::endl;
        return;
    }
    
    // Print header
    std::cout << std::setw(20) << std::left << "Variant"
              << std::setw(12) << std::right << "Executions"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(12) << "Total (ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Print metrics for each variant
    for (const auto& metrics : allMetrics) {
        std::cout << std::setw(20) << std::left << metrics.variantName
                  << std::setw(12) << std::right << metrics.numExecutions
                  << std::setw(12) << std::fixed << std::setprecision(2) << metrics.avgTime
                  << std::setw(12) << metrics.minTime
                  << std::setw(12) << metrics.maxTime
                  << std::setw(12) << metrics.totalTime << std::endl;
    }
    
    // Print speedup comparison (relative to slowest)
    if (allMetrics.size() > 1) {
        std::cout << "\n=== SPEEDUP COMPARISON (vs. slowest variant) ===" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double slowestTime = allMetrics.back().avgTime;
        
        for (const auto& metrics : allMetrics) {
            double speedup = slowestTime / metrics.avgTime;
            double efficiency = (speedup - 1.0) / speedup * 100.0;
            
            std::cout << std::setw(20) << std::left << metrics.variantName
                      << " Speedup: " << std::fixed << std::setprecision(2) << speedup << "x"
                      << " | Efficiency: " << std::setprecision(1) << efficiency << "%" << std::endl;
        }
    }
    
    std::cout << std::string(80, '=') << std::endl;
}

void PerformanceMeasurementOMP::saveVariantReport(const std::string& filepath) {
    std::ofstream reportFile(filepath);
    
    if (!reportFile.is_open()) {
        std::cerr << "Failed to create variant report file: " << filepath << std::endl;
        return;
    }
    
    reportFile << "CONVOLUTION VARIANT PERFORMANCE REPORT\n";
    reportFile << "======================================\n\n";
    
    auto allMetrics = getAllVariantMetrics();
    
    // Summary table
    reportFile << "Summary:\n";
    reportFile << std::setw(20) << std::left << "Variant"
               << std::setw(12) << std::right << "Executions"
               << std::setw(12) << "Avg (ms)"
               << std::setw(12) << "Min (ms)"
               << std::setw(12) << "Max (ms)"
               << std::setw(12) << "Total (ms)" << "\n";
    reportFile << std::string(80, '-') << "\n";
    
    for (const auto& metrics : allMetrics) {
        reportFile << std::setw(20) << std::left << metrics.variantName
                   << std::setw(12) << std::right << metrics.numExecutions
                   << std::setw(12) << std::fixed << std::setprecision(2) << metrics.avgTime
                   << std::setw(12) << metrics.minTime
                   << std::setw(12) << metrics.maxTime
                   << std::setw(12) << metrics.totalTime << "\n";
    }
    
    // Speedup analysis
    if (allMetrics.size() > 1) {
        reportFile << "\n\nSpeedup Analysis (relative to slowest variant):\n";
        reportFile << std::string(80, '-') << "\n";
        
        double slowestTime = allMetrics.back().avgTime;
        
        for (const auto& metrics : allMetrics) {
            double speedup = slowestTime / metrics.avgTime;
            reportFile << std::setw(20) << std::left << metrics.variantName
                       << " Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        
        // Best variant
        reportFile << "\n\nRecommendation:\n";
        reportFile << "Best performing variant: " << allMetrics[0].variantName 
                   << " (Avg: " << std::fixed << std::setprecision(2) 
                   << allMetrics[0].avgTime << " ms)\n";
    }
    
    reportFile.close();
    std::cout << "Variant comparison report saved to: " << filepath << std::endl;
}

void PerformanceMeasurementOMP::compareVariants(const std::vector<std::string>& variantNames) {
    std::cout << "\n=== DETAILED VARIANT COMPARISON ===" << std::endl;
    
    for (size_t i = 0; i < variantNames.size(); i++) {
        for (size_t j = i + 1; j < variantNames.size(); j++) {
            double speedup = getSpeedupBetweenVariants(variantNames[i], variantNames[j]);
            
            std::cout << variantNames[i] << " vs " << variantNames[j] 
                      << ": " << std::fixed << std::setprecision(2) << speedup << "x";
            
            if (speedup > 1.0) {
                std::cout << " (" << variantNames[i] << " is faster)";
            } else if (speedup < 1.0) {
                std::cout << " (" << variantNames[j] << " is faster)";
            } else {
                std::cout << " (similar performance)";
            }
            std::cout << std::endl;
        }
    }
}

double PerformanceMeasurementOMP::getSpeedupBetweenVariants(const std::string& baseVariant, 
                                                             const std::string& compareVariant) {
    auto baseMetrics = getVariantMetrics(baseVariant);
    auto compareMetrics = getVariantMetrics(compareVariant);
    
    if (baseMetrics.avgTime <= 0.0 || compareMetrics.avgTime <= 0.0) {
        return 0.0;
    }
    
    return compareMetrics.avgTime / baseMetrics.avgTime;
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

void PerformanceMeasurementOMP::printDetailedResults(const std::string& operationName, 
                                                      int numThreads, double serialTime) {
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
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_64F);
    diff = diff.mul(diff);
    cv::Scalar mse = cv::sum(diff);
    double total_mse = mse[0] + mse[1] + mse[2];
    return total_mse / (img1.rows * img1.cols * img1.channels());
}

double PerformanceMeasurementOMP::calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = calculateMSE(img1, img2);
    if (mse < 1e-10) {
        return 100.0;
    }
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}

bool PerformanceMeasurementOMP::validateResult(const cv::Mat& result, int expectedWidth, 
                                                 int expectedHeight, int expectedChannels) {
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

void PerformanceMeasurementOMP::analyzeThreadUtilization() {
    std::cout << "\n=== THREAD UTILIZATION ANALYSIS ===" << std::endl;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " of " << num_threads << " is active" << std::endl;
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
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Threads in current parallel region: " << omp_get_num_threads() << std::endl;
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

// did the changes to upload 