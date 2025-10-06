

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>
#include <filesystem>
#include <fstream>

#include "Header/gaussian_filter.h"
#include "Header/sobel_filter.h"
#include "Header/sharpening_filter.h"
#include "Header/laplacian_filter.h"
#include "Header/convolution_engine_omp.h"
#include "Header/performance_measure_omp.h"

namespace fs = std::filesystem;

enum FilterType {
    GAUSSIAN,
    SOBEL_X,
    SOBEL_Y,
    SOBEL_MAGNITUDE,
    SHARPENING,
    LAPLACIAN,
    EDGE_DETECTION
};

struct ImageProcessingResult {
    std::string imageName;
    std::string filterName;
    double processingTime;
    bool success;
};

class UCMercedBuildingsBenchmark {
private:
    PerformanceMeasurementOMP performance;
    std::vector<ImageProcessingResult> allResults;
    
    // Filter instances
    GaussianFilterOMP gaussianFilter;
    SobelFilterOMP sobelFilter;
    SharpeningFilterOMP sharpeningFilter;
    LaplacianFilterOMP laplacianFilter;
    
    std::string inputFolder;
    std::string outputFolder;
    int numThreads;

public:
    UCMercedBuildingsBenchmark(const std::string& inputPath, 
                               const std::string& outputPath,
                               int threads = omp_get_max_threads()) 
        : inputFolder(inputPath), outputFolder(outputPath), numThreads(threads) {
        
        gaussianFilter = GaussianFilterOMP();
        sobelFilter = SobelFilterOMP();
        sharpeningFilter = SharpeningFilterOMP();
        laplacianFilter = LaplacianFilterOMP();
        
        omp_set_num_threads(numThreads);
        
        // Create output directory structure
        createOutputDirectories();
    }

    void createOutputDirectories() {
        try {
            fs::create_directories(outputFolder);
            fs::create_directories(outputFolder + "/gaussian");
            fs::create_directories(outputFolder + "/sobel_x");
            fs::create_directories(outputFolder + "/sobel_y");
            fs::create_directories(outputFolder + "/sobel_magnitude");
            fs::create_directories(outputFolder + "/sharpening");
            fs::create_directories(outputFolder + "/laplacian");
            fs::create_directories(outputFolder + "/edge_detection");
            fs::create_directories(outputFolder + "/reports");
            
            std::cout << "Output directories created successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating directories: " << e.what() << std::endl;
        }
    }

    cv::Mat applyFilter(const cv::Mat& input, FilterType filterType) {
        cv::Mat result;
        
        switch(filterType) {
            case GAUSSIAN:
                result = gaussianFilter.apply(input, 5, 1.4);
                break;
            case SOBEL_X:
                result = sobelFilter.apply(input);
                break;
            case SOBEL_Y: {
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
                break;
            }
            case SOBEL_MAGNITUDE:
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
    
    std::string getFilterName(FilterType filterType) {
        switch(filterType) {
            case GAUSSIAN: return "gaussian";
            case SOBEL_X: return "sobel_x";
            case SOBEL_Y: return "sobel_y";
            case SOBEL_MAGNITUDE: return "sobel_magnitude";
            case SHARPENING: return "sharpening";
            case LAPLACIAN: return "laplacian";
            case EDGE_DETECTION: return "edge_detection";
            default: return "unknown";
        }
    }
    
    std::string getFilterDisplayName(FilterType filterType) {
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

    std::vector<std::string> getImageFiles() {
        std::vector<std::string> imageFiles;
        
        try {
            for (const auto& entry : fs::directory_iterator(inputFolder)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                        ext == ".tif" || ext == ".tiff" || ext == ".bmp") {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading directory: " << e.what() << std::endl;
        }
        
        std::sort(imageFiles.begin(), imageFiles.end());
        return imageFiles;
    }

    void processImage(const std::string& imagePath, FilterType filterType) {
        std::string imageName = fs::path(imagePath).stem().string();
        std::string filterName = getFilterName(filterType);
        std::string filterDisplayName = getFilterDisplayName(filterType);
        
        std::cout << "  Processing: " << imageName << " with " 
                  << filterDisplayName << "..." << std::flush;
        
        ImageProcessingResult result;
        result.imageName = imageName;
        result.filterName = filterDisplayName;
        result.success = false;
        
        try {
            cv::Mat image = cv::imread(imagePath);
            
            if (image.empty()) {
                std::cout << " FAILED (cannot load image)" << std::endl;
                result.processingTime = 0.0;
                allResults.push_back(result);
                return;
            }
            
            performance.startTimer();
            cv::Mat filtered = applyFilter(image, filterType);
            performance.stopTimer();
            
            result.processingTime = performance.getElapsedMilliseconds();
            
            // Save result
            cv::Mat finalResult;
            filtered.convertTo(finalResult, CV_8U);
            
            std::string outputPath = outputFolder + "/" + filterName + "/" + 
                                    imageName + "_" + filterName + ".jpg";
            cv::imwrite(outputPath, finalResult);
            
            result.success = true;
            std::cout << " OK (" << std::fixed << std::setprecision(2) 
                     << result.processingTime << " ms)" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << " FAILED (" << e.what() << ")" << std::endl;
            result.processingTime = 0.0;
        }
        
        allResults.push_back(result);
    }

    void processBatch() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "BATCH PROCESSING UCMerced BUILDINGS DATASET" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Input folder: " << inputFolder << std::endl;
        std::cout << "Output folder: " << outputFolder << std::endl;
        std::cout << "Threads: " << numThreads << std::endl;
        
        std::vector<std::string> imageFiles = getImageFiles();
        
        if (imageFiles.empty()) {
            std::cout << "\nNo image files found in: " << inputFolder << std::endl;
            return;
        }
        
        std::cout << "Found " << imageFiles.size() << " images" << std::endl;
        
        std::vector<FilterType> filters = {
            GAUSSIAN, SOBEL_X, SOBEL_Y, SOBEL_MAGNITUDE, 
            SHARPENING, LAPLACIAN, EDGE_DETECTION
        };
        
        for (FilterType filter : filters) {
            std::cout << "\n--- Applying " << getFilterDisplayName(filter) 
                     << " to all images ---" << std::endl;
            
            for (const auto& imagePath : imageFiles) {
                processImage(imagePath, filter);
            }
        }
    }

    void generateReport() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "GENERATING PERFORMANCE REPORT" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::string reportPath = outputFolder + "/reports/performance_report.txt";
        std::ofstream reportFile(reportPath);
        
        if (!reportFile.is_open()) {
            std::cerr << "Failed to create report file" << std::endl;
            return;
        }
        
        // Header
        reportFile << "UCMerced Buildings Dataset - Filter Performance Report\n";
        reportFile << "=====================================================\n\n";
        reportFile << "Configuration:\n";
        reportFile << "  Threads: " << numThreads << "\n";
        reportFile << "  Input folder: " << inputFolder << "\n";
        reportFile << "  Total images processed: " << allResults.size() / 7 << "\n\n";
        
        // Summary statistics per filter
        std::map<std::string, std::vector<double>> filterTimes;
        std::map<std::string, int> filterSuccessCount;
        
        for (const auto& result : allResults) {
            if (result.success) {
                filterTimes[result.filterName].push_back(result.processingTime);
                filterSuccessCount[result.filterName]++;
            }
        }
        
        reportFile << "Filter Performance Summary:\n";
        reportFile << "===========================\n\n";
        
        for (const auto& pair : filterTimes) {
            const std::string& filterName = pair.first;
            const std::vector<double>& times = pair.second;
            
            if (times.empty()) continue;
            
            double totalTime = 0.0;
            double minTime = times[0];
            double maxTime = times[0];
            
            for (double time : times) {
                totalTime += time;
                minTime = std::min(minTime, time);
                maxTime = std::max(maxTime, time);
            }
            
            double avgTime = totalTime / times.size();
            
            reportFile << filterName << ":\n";
            reportFile << "  Images processed: " << filterSuccessCount[filterName] << "\n";
            reportFile << "  Total time: " << std::fixed << std::setprecision(2) 
                      << totalTime << " ms\n";
            reportFile << "  Average time: " << avgTime << " ms\n";
            reportFile << "  Min time: " << minTime << " ms\n";
            reportFile << "  Max time: " << maxTime << " ms\n\n";
        }
        
        // Detailed results
        reportFile << "\nDetailed Results:\n";
        reportFile << "=================\n\n";
        reportFile << std::setw(30) << std::left << "Image Name"
                  << std::setw(20) << "Filter"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(10) << "Status" << "\n";
        reportFile << std::string(75, '-') << "\n";
        
        for (const auto& result : allResults) {
            reportFile << std::setw(30) << std::left << result.imageName
                      << std::setw(20) << result.filterName
                      << std::setw(15) << std::fixed << std::setprecision(2) 
                      << result.processingTime
                      << std::setw(10) << (result.success ? "OK" : "FAILED") << "\n";
        }
        
        reportFile.close();
        std::cout << "Report saved to: " << reportPath << std::endl;
        
        // Also print summary to console
        std::cout << "\n=== PROCESSING SUMMARY ===" << std::endl;
        for (const auto& pair : filterTimes) {
            const std::string& filterName = pair.first;
            const std::vector<double>& times = pair.second;
            
            if (times.empty()) continue;
            
            double totalTime = 0.0;
            for (double time : times) {
                totalTime += time;
            }
            double avgTime = totalTime / times.size();
            
            std::cout << std::setw(20) << std::left << filterName 
                     << ": " << filterSuccessCount[filterName] << " images, "
                     << "Avg: " << std::fixed << std::setprecision(2) 
                     << avgTime << " ms" << std::endl;
        }
    }

    void generateCSVReport() {
        std::string csvPath = outputFolder + "/reports/performance_data.csv";
        std::ofstream csvFile(csvPath);
        
        if (!csvFile.is_open()) {
            std::cerr << "Failed to create CSV file" << std::endl;
            return;
        }
        
        csvFile << "ImageName,FilterName,ProcessingTime_ms,Success\n";
        
        for (const auto& result : allResults) {
            csvFile << result.imageName << ","
                   << result.filterName << ","
                   << std::fixed << std::setprecision(3) << result.processingTime << ","
                   << (result.success ? "true" : "false") << "\n";
        }
        
        csvFile.close();
        std::cout << "CSV report saved to: " << csvPath << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "UCMerced BUILDINGS DATASET FILTER BENCHMARK" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Default paths - modify these as needed
    std::string inputFolder = "C:\\Users\\DELL\\Downloads\\UCMerced_LandUse\\UCMerced_LandUse\\Images\\buildings";
    std::string outputFolder = "C:\\Users\\DELL\\hpc project\\Team-B5-HPC\\UCMerced_Output_Buildings";
    int numThreads = omp_get_max_threads();

    // Parse command line arguments
    if (argc > 1) inputFolder = argv[1];
    if (argc > 2) outputFolder = argv[2];
    if (argc > 3) numThreads = std::stoi(argv[3]);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input folder: " << inputFolder << std::endl;
    std::cout << "  Output folder: " << outputFolder << std::endl;
    std::cout << "  Maximum threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "  Using threads: " << numThreads << std::endl;

    // Check if input folder exists
    if (!fs::exists(inputFolder)) {
        std::cerr << "\nERROR: Input folder does not exist!" << std::endl;
        std::cerr << "Please check the path: " << inputFolder << std::endl;
        return 1;
    }

    // Create benchmark instance and run
    UCMercedBuildingsBenchmark benchmark(inputFolder, outputFolder, numThreads);
    
    // Process all images
    benchmark.processBatch();
    
    // Generate reports
    benchmark.generateReport();
    benchmark.generateCSVReport();

    std::cout << "\n========================================" << std::endl;
    std::cout << "PROCESSING COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nFiltered images saved in: " << outputFolder << std::endl;
    std::cout << "Performance reports saved in: " << outputFolder << "\\reports" << std::endl;

    return 0;
}