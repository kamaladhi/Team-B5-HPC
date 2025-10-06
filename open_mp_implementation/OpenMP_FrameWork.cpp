// for a single image we made a framework

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

#include "Header/gaussian_filter.h"
#include "Header/sobel_filter.h"
#include "Header/sharpening_filter.h"
#include "Header/laplacian_filter.h"
#include "Header/convolution_engine_omp.h"
#include "Header/performance_measure_omp.h"
enum FilterType {
    GAUSSIAN,
    SOBEL_X,
    SOBEL_Y,
    SOBEL_MAGNITUDE,
    SHARPENING,
    LAPLACIAN,
    EDGE_DETECTION
};


enum ConvolutionMethod {
    BASIC = 0,
    BALANCED = 1,
    CACHE_OPTIMIZED = 2
};

struct BenchmarkResult {
    std::string filterName;
    std::string methodName;
    int threads;
    double executionTime;
    double speedup;
    double efficiency;
};

class ComprehensiveFilterBenchmark {
private:
    PerformanceMeasurementOMP performance;
    std::vector<BenchmarkResult> allResults;
    
    // Filter instances
    GaussianFilterOMP gaussianFilter;
    SobelFilterOMP sobelFilter;
    SharpeningFilterOMP sharpeningFilter;
    LaplacianFilterOMP laplacianFilter;

public:
    ComprehensiveFilterBenchmark() {
        // Default constructors only
        gaussianFilter = GaussianFilterOMP();
        sobelFilter = SobelFilterOMP();
        sharpeningFilter = SharpeningFilterOMP();
        laplacianFilter = LaplacianFilterOMP();
    }

    // Apply filter based on type
    cv::Mat applyFilter(const cv::Mat& input, FilterType filterType) {
        cv::Mat result;
        
        switch(filterType) {
            case GAUSSIAN:
                // GaussianFilterOMP::apply requires kernelSize and sigma parameters
                result = gaussianFilter.apply(input, 5, 1.4);
                break;
            case SOBEL_X:
                // SobelFilterOMP::apply returns Sobel X by default
                result = sobelFilter.apply(input);
                break;
            case SOBEL_Y:
                // For Sobel Y, we'll use apply and assume it handles Y direction
                // Or use the raw convolution with Sobel Y kernel
                {
                    std::vector<std::vector<float>> sobelY = {
                        {-1, -2, -1},
                        { 0,  0,  0},
                        { 1,  2,  1}
                    };
                    cv::Mat grayInput;
                    if (input.channels() == 3) {
                        cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
                    } else {
                        grayInput = input.clone();
                    }
                    result = ConvolutionEngineOMP::convolve2D(grayInput, sobelY);
                }
                break;
            case SOBEL_MAGNITUDE:
                // Use apply method which should give magnitude
                result = sobelFilter.apply(input);
                break;
            case SHARPENING:
                result = sharpeningFilter.apply(input);
                break;
            case LAPLACIAN:
                result = laplacianFilter.apply(input);
                break;
            case EDGE_DETECTION: {
                cv::Mat grayInput;
                if (input.channels() == 3) {
                    cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
                } else {
                    grayInput = input.clone();
                }
                std::vector<std::vector<float>> edgeKernel = {
                    {-1, -1, -1},
                    {-1,  8, -1},
                    {-1, -1, -1}
                };
                result = ConvolutionEngineOMP::convolve2D(grayInput, edgeKernel);
                break;
            }
        }
        
        return result;
    }
    
    // Get filter name
    std::string getFilterName(FilterType filterType) {
        switch(filterType) {
            case GAUSSIAN: return "Gaussian Blur";
            case SOBEL_X: return "Sobel X";
            case SOBEL_Y: return "Sobel Y";
            case SOBEL_MAGNITUDE: return "Sobel Magnitude";
            case SHARPENING: return "Sharpening";
            case LAPLACIAN: return "Laplacian";
            case EDGE_DETECTION: return "Edge Detection";
            default: return "Unknown";
        }
    }
    
    // Benchmark a single filter with all convolution methods
    void benchmarkFilter(const cv::Mat& input, FilterType filterType, 
                        int numThreads, const std::string& outputPath) {
        std::string filterName = getFilterName(filterType);
        std::cout << "\n========================================" << std::endl;
        std::cout << "BENCHMARKING: " << filterName << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
        std::cout << "========================================" << std::endl;
        
        omp_set_num_threads(numThreads);
        
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        std::vector<std::string> methodNames = {
            "Basic",
            "Balanced", 
            "Cache-Optimized"
        };
        
        std::vector<double> executionTimes;
        std::vector<cv::Mat> results;
        
        // Test using the actual filter implementation
        std::cout << "\n--- Using Native Filter Implementation ---" << std::endl;
        performance.startTimer();
        cv::Mat nativeResult = applyFilter(grayInput, filterType);
        performance.stopTimer();
        double nativeTime = performance.getElapsedMilliseconds();
        executionTimes.push_back(nativeTime);
        results.push_back(nativeResult);
        
        std::cout << "Execution time: " << std::fixed << std::setprecision(3) 
                  << nativeTime << " ms" << std::endl;
        
        // Save native result
        cv::Mat finalNative;
        nativeResult.convertTo(finalNative, CV_8U);
        std::string filename = outputPath + "_" + filterName + "_native_" 
                             + std::to_string(numThreads) + "t.jpg";
        cv::imwrite(filename, finalNative);
        std::cout << "Saved: " << filename << std::endl;
        
        // Store result
        BenchmarkResult result;
        result.filterName = filterName;
        result.methodName = "Native";
        result.threads = numThreads;
        result.executionTime = nativeTime;
        result.speedup = 1.0;
        result.efficiency = 1.0;
        allResults.push_back(result);
        
        // Test with raw convolution methods (if applicable)
        if (filterType == EDGE_DETECTION || filterType == LAPLACIAN) {
            std::vector<std::vector<float>> kernel;
            
            if (filterType == EDGE_DETECTION) {
                kernel = {
                    {-1, -1, -1},
                    {-1,  8, -1},
                    {-1, -1, -1}
                };
            } else if (filterType == LAPLACIAN) {
                kernel = {
                    { 0, -1,  0},
                    {-1,  4, -1},
                    { 0, -1,  0}
                };
            }
            
            // Test all three convolution methods
            for (int method = 0; method < 3; method++) {
                std::cout << "\n--- Testing: " << methodNames[method] << " Convolution ---" << std::endl;
                
                performance.startTimer();
                cv::Mat methodResult;
                
                switch(method) {
                    case BASIC:
                        methodResult = ConvolutionEngineOMP::convolve2D(grayInput, kernel);
                        break;
                    case BALANCED:
                        methodResult = ConvolutionEngineOMP::convolve2DBalanced(grayInput, kernel);
                        break;
                    case CACHE_OPTIMIZED:
                        methodResult = ConvolutionEngineOMP::convolve2DCacheOptimized(grayInput, kernel);
                        break;
                }
                
                performance.stopTimer();
                double execTime = performance.getElapsedMilliseconds();
                executionTimes.push_back(execTime);
                results.push_back(methodResult);
                
                std::cout << "Execution time: " << std::fixed << std::setprecision(3) 
                          << execTime << " ms" << std::endl;
                
                // Save result
                cv::Mat finalResult;
                methodResult.convertTo(finalResult, CV_8U);
                std::string filename = outputPath + "_" + filterName + "_" 
                                     + methodNames[method] + "_" 
                                     + std::to_string(numThreads) + "t.jpg";
                cv::imwrite(filename, finalResult);
                std::cout << "Saved: " << filename << std::endl;
                
                // Store result
                BenchmarkResult br;
                br.filterName = filterName;
                br.methodName = methodNames[method];
                br.threads = numThreads;
                br.executionTime = execTime;
                br.speedup = nativeTime / execTime;
                br.efficiency = br.speedup;
                allResults.push_back(br);
            }
        }
        
        // Print comparison
        std::cout << "\n=== PERFORMANCE COMPARISON ===" << std::endl;
        std::cout << std::setw(20) << "Method" << std::setw(15) << "Time (ms)" 
                  << std::setw(15) << "Relative" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        double fastestTime = *std::min_element(executionTimes.begin(), executionTimes.end());
        
        std::cout << std::setw(20) << "Native" 
                  << std::setw(15) << std::fixed << std::setprecision(3) << nativeTime
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << (nativeTime / fastestTime) << "x" << std::endl;
        
        if (executionTimes.size() > 1) {
            for (size_t i = 1; i < executionTimes.size(); i++) {
                std::cout << std::setw(20) << methodNames[i-1] 
                          << std::setw(15) << std::fixed << std::setprecision(3) << executionTimes[i]
                          << std::setw(15) << std::fixed << std::setprecision(2) 
                          << (executionTimes[i] / fastestTime) << "x" << std::endl;
            }
        }
    }
    
    // Run thread scalability analysis for all filters
    void runThreadScalabilityAnalysis(const cv::Mat& input, 
                                     const std::vector<int>& threadCounts) {
        std::cout << "\n\n========================================" << std::endl;
        std::cout << "THREAD SCALABILITY ANALYSIS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Maximum threads available: " << omp_get_max_threads() << std::endl;
        
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        // Test all filters
        std::vector<FilterType> filters = {
            GAUSSIAN, SOBEL_MAGNITUDE, SHARPENING, LAPLACIAN, EDGE_DETECTION
        };
        
        std::map<FilterType, std::vector<double>> filterTimes;
        
        for (FilterType filter : filters) {
            std::string filterName = getFilterName(filter);
            std::cout << "\n--- Analyzing: " << filterName << " ---" << std::endl;
            
            for (int threads : threadCounts) {
                if (threads <= omp_get_max_threads()) {
                    omp_set_num_threads(threads);
                    performance.startTimer();
                    
                    cv::Mat result = applyFilter(grayInput, filter);
                    
                    performance.stopTimer();
                    double execTime = performance.getElapsedMilliseconds();
                    filterTimes[filter].push_back(execTime);
                    
                    std::cout << "Threads: " << std::setw(2) << threads 
                              << ", Time: " << std::fixed << std::setprecision(2) 
                              << execTime << " ms" << std::endl;
                }
            }
        }
        
        // Print speedup comparison table
        std::cout << "\n\n=== SPEEDUP COMPARISON TABLE ===" << std::endl;
        std::cout << std::setw(20) << "Filter";
        for (int threads : threadCounts) {
            if (threads <= omp_get_max_threads()) {
                std::cout << std::setw(12) << (std::to_string(threads) + "T");
            }
        }
        std::cout << std::endl;
        std::cout << std::string(20 + 12 * threadCounts.size(), '-') << std::endl;
        
        for (FilterType filter : filters) {
            std::string filterName = getFilterName(filter);
            std::cout << std::setw(20) << filterName;
            
            const auto& times = filterTimes[filter];
            for (size_t i = 0; i < times.size(); i++) {
                double speedup = times[0] / times[i];
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) 
                          << speedup << "x";
            }
            std::cout << std::endl;
        }
        
        // Print efficiency comparison table
        std::cout << "\n=== EFFICIENCY COMPARISON TABLE ===" << std::endl;
        std::cout << std::setw(20) << "Filter";
        for (int threads : threadCounts) {
            if (threads <= omp_get_max_threads()) {
                std::cout << std::setw(12) << (std::to_string(threads) + "T");
            }
        }
        std::cout << std::endl;
        std::cout << std::string(20 + 12 * threadCounts.size(), '-') << std::endl;
        
        for (FilterType filter : filters) {
            std::string filterName = getFilterName(filter);
            std::cout << std::setw(20) << filterName;
            
            const auto& times = filterTimes[filter];
            for (size_t i = 0; i < times.size(); i++) {
                double speedup = times[0] / times[i];
                double efficiency = (speedup / threadCounts[i]) * 100;
                std::cout << std::setw(12) << std::fixed << std::setprecision(1) 
                          << efficiency << "%";
            }
            std::cout << std::endl;
        }
        
        // Print execution time comparison table
        std::cout << "\n=== EXECUTION TIME COMPARISON (ms) ===" << std::endl;
        std::cout << std::setw(20) << "Filter";
        for (int threads : threadCounts) {
            if (threads <= omp_get_max_threads()) {
                std::cout << std::setw(12) << (std::to_string(threads) + "T");
            }
        }
        std::cout << std::endl;
        std::cout << std::string(20 + 12 * threadCounts.size(), '-') << std::endl;
        
        for (FilterType filter : filters) {
            std::string filterName = getFilterName(filter);
            std::cout << std::setw(20) << filterName;
            
            const auto& times = filterTimes[filter];
            for (double time : times) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) << time;
            }
            std::cout << std::endl;
        }
    }
    
    // Compare all convolution methods for a specific kernel
    void compareConvolutionMethods(const cv::Mat& input, 
                                  const std::vector<std::vector<float>>& kernel,
                                  const std::string& kernelName,
                                  int numThreads) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "CONVOLUTION METHOD COMPARISON" << std::endl;
        std::cout << "Kernel: " << kernelName << std::endl;
        std::cout << "========================================" << std::endl;
        
        omp_set_num_threads(numThreads);
        
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        std::vector<std::string> methodNames = {
            "Basic", "Balanced", "Cache-Optimized"
        };
        
        std::vector<double> times;
        
        for (int method = 0; method < 3; method++) {
            std::cout << "\n--- " << methodNames[method] << " ---" << std::endl;
            
            performance.startTimer();
            cv::Mat result;
            
            switch(method) {
                case BASIC:
                    result = ConvolutionEngineOMP::convolve2D(grayInput, kernel);
                    break;
                case BALANCED:
                    result = ConvolutionEngineOMP::convolve2DBalanced(grayInput, kernel);
                    break;
                case CACHE_OPTIMIZED:
                    result = ConvolutionEngineOMP::convolve2DCacheOptimized(grayInput, kernel);
                    break;
            }
            
            performance.stopTimer();
            double execTime = performance.getElapsedMilliseconds();
            times.push_back(execTime);
            
            std::cout << "Time: " << std::fixed << std::setprecision(3) 
                      << execTime << " ms" << std::endl;
        }
        
        std::cout << "\n=== METHOD COMPARISON ===" << std::endl;
        double fastestTime = *std::min_element(times.begin(), times.end());
        
        for (size_t i = 0; i < methodNames.size(); i++) {
            std::cout << std::setw(20) << methodNames[i]
                      << ": " << std::setw(10) << std::fixed << std::setprecision(3) 
                      << times[i] << " ms ("
                      << std::fixed << std::setprecision(2) 
                      << (times[i] / fastestTime) << "x)" << std::endl;
        }
    }
    
    // Run comprehensive benchmark
    void runComprehensiveBenchmark(const cv::Mat& input, int numThreads, 
                                  const std::string& outputPath) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "COMPREHENSIVE FILTER BENCHMARK" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Image size: " << input.cols << "x" << input.rows << std::endl;
        std::cout << "Channels: " << input.channels() << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
        
        // Benchmark all filters
        std::vector<FilterType> filters = {
            GAUSSIAN, SOBEL_X, SOBEL_Y, SOBEL_MAGNITUDE, 
            SHARPENING, LAPLACIAN, EDGE_DETECTION
        };
        
        for (FilterType filter : filters) {
            benchmarkFilter(input, filter, numThreads, outputPath);
        }
        
        // Test raw convolution with different methods
        std::vector<std::vector<float>> testKernels[] = {
            {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}},  // Edge
            {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}},      // Laplacian
            {{0.111f, 0.111f, 0.111f}, {0.111f, 0.111f, 0.111f}, {0.111f, 0.111f, 0.111f}}  // Blur
        };
        
        std::string kernelNames[] = {"Edge", "Laplacian", "Blur"};
        
        for (int i = 0; i < 3; i++) {
            compareConvolutionMethods(input, testKernels[i], kernelNames[i], numThreads);
        }
    }
    
    // Print final summary
    void printFinalSummary() {
        std::cout << "\n\n========================================" << std::endl;
        std::cout << "FINAL BENCHMARK SUMMARY" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total tests run: " << allResults.size() << std::endl;
        
        // Find best performing filter
        auto fastest = std::min_element(allResults.begin(), allResults.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.executionTime < b.executionTime;
            });
        
        if (fastest != allResults.end()) {
            std::cout << "\nFastest execution:" << std::endl;
            std::cout << "  Filter: " << fastest->filterName << std::endl;
            std::cout << "  Method: " << fastest->methodName << std::endl;
            std::cout << "  Time: " << std::fixed << std::setprecision(3) 
                      << fastest->executionTime << " ms" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "COMPREHENSIVE OPENMP FILTER COMPARISON" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::string imagePath = (argc > 1) ? argv[1] : "test2.jpg";
    int numThreads = (argc > 2) ? std::stoi(argv[2]) : omp_get_max_threads();

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Maximum threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "  Using threads: " << numThreads << std::endl;
    std::cout << "  Input image: " << imagePath << std::endl;

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "\nCreating synthetic test image..." << std::endl;
        image = cv::Mat::zeros(1024, 1024, CV_8UC3);
        cv::rectangle(image, cv::Point(100, 100), cv::Point(924, 924), 
                     cv::Scalar(255, 255, 255), -1);
        cv::circle(image, cv::Point(512, 512), 200, cv::Scalar(128, 128, 128), -1);
        for (int i = 0; i < 5000; i++) {
            int x = rand() % image.cols;
            int y = rand() % image.rows;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        }
        cv::imwrite("synthetic_test.jpg", image);
        std::cout << "Synthetic test image created: synthetic_test.jpg" << std::endl;
    } else {
        std::cout << "Loaded image: " << imagePath << std::endl;
    }

    ComprehensiveFilterBenchmark benchmark;

    // Run comprehensive benchmark with specified thread count
    benchmark.runComprehensiveBenchmark(image, numThreads, "output");

    // Run thread scalability analysis
    std::vector<int> threadCounts = {1, 2, 4, 8, 12, 16};
    benchmark.runThreadScalabilityAnalysis(image, threadCounts);

    // Print final summary
    benchmark.printFinalSummary();

    std::cout << "\n========================================" << std::endl;
    std::cout << "PROCESSING COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nAll results have been saved with appropriate suffixes." << std::endl;
    std::cout << "Check the console output for detailed performance metrics." << std::endl;

    return 0;
}