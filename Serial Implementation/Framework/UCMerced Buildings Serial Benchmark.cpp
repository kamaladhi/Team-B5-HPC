#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <map>

#ifdef _WIN32
    #include <windows.h>
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif

#include "convolution_engine.h"
#include "performance_measure.h"
#include "gaussian_filter.h"
#include "sobel_filter.h"
#include "sharpening_filter.h"
#include "laplacian_filter.h"

enum FilterType {
    GAUSSIAN,
    SOBEL_X,
    SOBEL_Y,
    SOBEL_MAGNITUDE,
    SHARPENING,
    LAPLACIAN,
    EDGE_DETECTION
};

enum ConvolutionVariant {
    STANDARD_MAT,
    RAW_ARRAY
};

struct ImageProcessingResult {
    std::string imageName;
    std::string filterName;
    std::string variantName;
    double processingTime;
    bool success;
};

class UCMercedBuildingsBenchmark {
private:
    PerformanceMeasurement performance;
    std::vector<ImageProcessingResult> allResults;
    
    // Filter instances
    GaussianFilterSerial gaussianFilter;
    SobelFilterSerial sobelFilter;
    SharpeningFilterSerial sharpeningFilter;
    LaplacianFilterSerial laplacianFilter;
    
    std::string inputFolder;
    std::string outputFolder;

public:
    UCMercedBuildingsBenchmark(const std::string& inputPath, 
                               const std::string& outputPath) 
        : inputFolder(inputPath), outputFolder(outputPath) {
        
        // Create output directory structure
        createOutputDirectories();
    }

    void createOutputDirectories() {
        try {
            fs::create_directories(outputFolder);
            
            // Create directories for each variant
            std::vector<std::string> variants = {"standard_mat", "raw_array"};
            std::vector<std::string> filters = {"gaussian", "sobel_x", "sobel_y", "sobel_magnitude", 
                                               "sharpening", "laplacian", "edge_detection"};
            
            for (const auto& variant : variants) {
                for (const auto& filter : filters) {
                    fs::create_directories(outputFolder + "/" + variant + "/" + filter);
                }
            }
            
            fs::create_directories(outputFolder + "/reports");
            
            std::cout << "Output directories created successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating directories: " << e.what() << std::endl;
        }
    }

    cv::Mat applyFilter(const cv::Mat& input, FilterType filterType, ConvolutionVariant variant) {
        cv::Mat result;
        cv::Mat grayInput;
        
        // Convert to grayscale if needed for certain filters
        if (input.channels() == 3 && (filterType == SOBEL_X || filterType == SOBEL_Y || 
                                     filterType == LAPLACIAN || filterType == EDGE_DETECTION)) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        switch(filterType) {
            case GAUSSIAN: {
                result = gaussianFilter.apply(input, 5, 1.4);
                break;
            }
            case SOBEL_X: {
                std::vector<std::vector<float>> sobelX = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
                };
                result = applyConvolutionVariant(grayInput, sobelX, variant);
                break;
            }
            case SOBEL_Y: {
                std::vector<std::vector<float>> sobelY = {
                    {-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}
                };
                result = applyConvolutionVariant(grayInput, sobelY, variant);
                break;
            }
            case SOBEL_MAGNITUDE: {
                result = sobelFilter.apply(input);
                break;
            }
            case SHARPENING: {
                result = sharpeningFilter.apply(input);
                break;
            }
            case LAPLACIAN: {
                std::vector<std::vector<float>> laplacianKernel = {
                    {0,  1, 0},
                    {1, -4, 1},
                    {0,  1, 0}
                };
                result = applyConvolutionVariant(grayInput, laplacianKernel, variant);
                break;
            }
            case EDGE_DETECTION: {
                std::vector<std::vector<float>> edgeKernel = {
                    {-1, -1, -1},
                    {-1,  8, -1},
                    {-1, -1, -1}
                };
                result = applyConvolutionVariant(grayInput, edgeKernel, variant);
                break;
            }
        }
        
        return result;
    }
    
    cv::Mat applyConvolutionVariant(const cv::Mat& input, 
                                    const std::vector<std::vector<float>>& kernel, 
                                    ConvolutionVariant variant) {
        cv::Mat result;
        
        switch(variant) {
            case STANDARD_MAT: {
                result = ConvolutionEngine::convolve2D(input, kernel);
                break;
            }
            case RAW_ARRAY: {
                // Convert to grayscale if not already
                cv::Mat grayInput;
                if (input.channels() == 3) {
                    cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
                } else {
                    grayInput = input.clone();
                }
                
                int width = grayInput.cols;
                int height = grayInput.rows;
                int kernelSize = kernel.size();
                
                // Flatten kernel
                std::vector<float> flatKernel;
                for (const auto& row : kernel) {
                    flatKernel.insert(flatKernel.end(), row.begin(), row.end());
                }
                
                // Prepare input and output arrays
                float* inputArray = new float[width * height];
                float* outputArray = new float[width * height];
                
                // Copy data to array
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        inputArray[y * width + x] = static_cast<float>(grayInput.at<uchar>(y, x));
                    }
                }
                
                // Call raw array convolution
                ConvolutionEngine::convolve2D(inputArray, outputArray, width, height, 
                                            flatKernel.data(), kernelSize);
                
                // Convert back to Mat
                result = cv::Mat(height, width, CV_32F);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        result.at<float>(y, x) = outputArray[y * width + x];
                    }
                }
                
                delete[] inputArray;
                delete[] outputArray;
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
    
    std::string getVariantName(ConvolutionVariant variant) {
        switch(variant) {
            case STANDARD_MAT: return "standard_mat";
            case RAW_ARRAY: return "raw_array";
            default: return "unknown";
        }
    }
    
    std::string getVariantDisplayName(ConvolutionVariant variant) {
        switch(variant) {
            case STANDARD_MAT: return "Standard Mat";
            case RAW_ARRAY: return "Raw Array";
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

    void processImage(const std::string& imagePath, FilterType filterType, ConvolutionVariant variant) {
        std::string imageName = fs::path(imagePath).stem().string();
        std::string filterName = getFilterName(filterType);
        std::string variantName = getVariantName(variant);
        std::string filterDisplayName = getFilterDisplayName(filterType);
        std::string variantDisplayName = getVariantDisplayName(variant);
        
        std::cout << "  Processing: " << imageName << " | " 
                  << filterDisplayName << " | " << variantDisplayName << "..." << std::flush;
        
        ImageProcessingResult result;
        result.imageName = imageName;
        result.filterName = filterDisplayName;
        result.variantName = variantDisplayName;
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
            cv::Mat filtered = applyFilter(image, filterType, variant);
            performance.stopTimer();
            
            result.processingTime = performance.getElapsedMilliseconds();
            
            // Save result
            cv::Mat finalResult;
            filtered.convertTo(finalResult, CV_8U);
            
            std::string outputPath = outputFolder + "/" + variantName + "/" + filterName + "/" + 
                                    imageName + "_" + filterName + "_" + variantName + ".jpg";
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
        std::cout << "SERIAL IMPLEMENTATION" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Input folder: " << inputFolder << std::endl;
        std::cout << "Output folder: " << outputFolder << std::endl;
        
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
        
        std::vector<ConvolutionVariant> variants = {
            STANDARD_MAT, RAW_ARRAY
        };
        
        // Process each filter with each variant
        for (FilterType filter : filters) {
            std::cout << "\n=== " << getFilterDisplayName(filter) << " ===" << std::endl;
            
            for (ConvolutionVariant variant : variants) {
                std::cout << "\n--- Variant: " << getVariantDisplayName(variant) << " ---" << std::endl;
                
                for (const auto& imagePath : imageFiles) {
                    processImage(imagePath, filter, variant);
                }
            }
        }
    }

    void generateReport() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "GENERATING PERFORMANCE REPORT" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::string reportPath = outputFolder + "/reports/serial_performance_report.txt";
        std::ofstream reportFile(reportPath);
        
        if (!reportFile.is_open()) {
            std::cerr << "Failed to create report file" << std::endl;
            return;
        }
        
        // Header
        reportFile << "UCMerced Buildings Dataset - Serial Filter Performance Report\n";
        reportFile << "============================================================\n\n";
        reportFile << "Configuration:\n";
        reportFile << "  Implementation: Serial (Single-threaded)\n";
        reportFile << "  Input folder: " << inputFolder << "\n";
        reportFile << "  Total images processed: " << allResults.size() / 14 << "\n\n";
        
        // Summary statistics per filter and variant
        std::map<std::pair<std::string, std::string>, std::vector<double>> filterVariantTimes;
        std::map<std::pair<std::string, std::string>, int> filterVariantSuccessCount;
        
        for (const auto& result : allResults) {
            if (result.success) {
                auto key = std::make_pair(result.filterName, result.variantName);
                filterVariantTimes[key].push_back(result.processingTime);
                filterVariantSuccessCount[key]++;
            }
        }
        
        reportFile << "Filter Performance Summary by Variant:\n";
        reportFile << "======================================\n\n";
        
        for (const auto& pair : filterVariantTimes) {
            const std::string& filterName = pair.first.first;
            const std::string& variantName = pair.first.second;
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
            
            reportFile << filterName << " - " << variantName << ":\n";
            reportFile << "  Images processed: " << filterVariantSuccessCount[pair.first] << "\n";
            reportFile << "  Total time: " << std::fixed << std::setprecision(2) 
                      << totalTime << " ms\n";
            reportFile << "  Average time: " << avgTime << " ms\n";
            reportFile << "  Min time: " << minTime << " ms\n";
            reportFile << "  Max time: " << maxTime << " ms\n\n";
        }
        
        // Variant comparison
        reportFile << "\nVariant Performance Comparison:\n";
        reportFile << "===============================\n\n";
        
        std::map<std::string, std::vector<double>> variantTotalTimes;
        for (const auto& pair : filterVariantTimes) {
            const std::string& variantName = pair.first.second;
            const std::vector<double>& times = pair.second;
            
            double totalTime = 0.0;
            for (double time : times) {
                totalTime += time;
            }
            variantTotalTimes[variantName].push_back(totalTime / times.size());
        }
        
        for (const auto& pair : variantTotalTimes) {
            double avgTime = 0.0;
            for (double time : pair.second) {
                avgTime += time;
            }
            avgTime /= pair.second.size();
            
            reportFile << pair.first << " - Overall Average: " 
                      << std::fixed << std::setprecision(2) << avgTime << " ms\n";
        }
        
        // Detailed results
        reportFile << "\n\nDetailed Results:\n";
        reportFile << "=================\n\n";
        reportFile << std::setw(25) << std::left << "Image Name"
                  << std::setw(20) << "Filter"
                  << std::setw(18) << "Variant"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(10) << "Status" << "\n";
        reportFile << std::string(88, '-') << "\n";
        
        for (const auto& result : allResults) {
            reportFile << std::setw(25) << std::left << result.imageName
                      << std::setw(20) << result.filterName
                      << std::setw(18) << result.variantName
                      << std::setw(15) << std::fixed << std::setprecision(2) 
                      << result.processingTime
                      << std::setw(10) << (result.success ? "OK" : "FAILED") << "\n";
        }
        
        reportFile.close();
        std::cout << "Report saved to: " << reportPath << std::endl;
        
        // Print summary to console
        std::cout << "\n=== VARIANT COMPARISON SUMMARY ===" << std::endl;
        for (const auto& pair : variantTotalTimes) {
            double avgTime = 0.0;
            for (double time : pair.second) {
                avgTime += time;
            }
            avgTime /= pair.second.size();
            
            std::cout << std::setw(20) << std::left << pair.first 
                     << ": Overall Avg = " << std::fixed << std::setprecision(2) 
                     << avgTime << " ms" << std::endl;
        }
    }

    void generateCSVReport() {
        std::string csvPath = outputFolder + "/reports/serial_performance_data.csv";
        std::ofstream csvFile(csvPath);
        
        if (!csvFile.is_open()) {
            std::cerr << "Failed to create CSV file" << std::endl;
            return;
        }
        
        csvFile << "ImageName,FilterName,VariantName,ProcessingTime_ms,Success\n";
        
        for (const auto& result : allResults) {
            csvFile << result.imageName << ","
                   << result.filterName << ","
                   << result.variantName << ","
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
    std::cout << "SERIAL IMPLEMENTATION" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Default paths
    std::string inputFolder = "C:\\Users\\DELL\\Downloads\\UCMerced_LandUse\\UCMerced_LandUse\\Images\\buildings";
    std::string outputFolder = "C:\\Users\\DELL\\hpc project\\Team-B5-HPC\\Serial Implementation\\UCMerced_Output_Buildings";

    // Parse command line arguments
    if (argc > 1) inputFolder = argv[1];
    if (argc > 2) outputFolder = argv[2];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input folder: " << inputFolder << std::endl;
    std::cout << "  Output folder: " << outputFolder << std::endl;
    std::cout << "  Implementation: Serial (Single-threaded)" << std::endl;

    // Check if input folder exists
    if (!fs::exists(inputFolder)) {
        std::cerr << "\nERROR: Input folder does not exist!" << std::endl;
        std::cerr << "Please check the path: " << inputFolder << std::endl;
        return 1;
    }

    // Create benchmark instance and run
    UCMercedBuildingsBenchmark benchmark(inputFolder, outputFolder);
    
    // Process all images with all variants
    benchmark.processBatch();
    
    // Generate reports
    benchmark.generateReport();
    benchmark.generateCSVReport();

    std::cout << "\n========================================" << std::endl;
    std::cout << "PROCESSING COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nFiltered images saved in: " << outputFolder << std::endl;
    std::cout << "Performance reports saved in: " << outputFolder << "\\reports" << std::endl;
    std::cout << "\nVariants tested:" << std::endl;
    std::cout << "  1. Standard Mat Convolution" << std::endl;
    std::cout << "  2. Raw Array Convolution" << std::endl;

    return 0;
}