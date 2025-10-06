#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>
#include <filesystem>
#include <fstream>

#include "gaussian_filter.h"
#include "sobel_filter.h"
#include "sharpening_filter.h"
#include "laplacian_filter.h"
#include "convolution_engine_omp.h"
#include "performance_measure_omp.h"

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

enum ConvolutionVariant {
    STANDARD,
    BALANCED,
    CACHE_OPTIMIZED,
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
    // constructor 
    UCMercedBuildingsBenchmark(const std::string& inputPath, const std::string& 
        outputPath,int threads = omp_get_max_threads()): inputFolder(inputPath), outputFolder(outputPath), numThreads(threads) {
        
        gaussianFilter = GaussianFilterOMP();
        sobelFilter = SobelFilterOMP();
        sharpeningFilter = SharpeningFilterOMP();
        laplacianFilter = LaplacianFilterOMP();
        
        omp_set_num_threads(numThreads);
        
        // Create output directory structure
        createOutputDirectories();
    }
    // function for creating a output directory 
    void createOutputDirectories() {
        try {
            fs::create_directories(outputFolder);
            
            // Create directories for each variant
            std::vector<std::string> variants = {"standard", "balanced", "cache_optimized", "raw_array"};
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
    // Function which is used for implementing a specific filters to an input image
    // takes image , filter type and convolution varient
    cv::Mat applyFilter(const cv::Mat& input, FilterType filterType, ConvolutionVariant variant) {
        cv::Mat result;
        cv::Mat grayInput;
        
        // Convert to grayscale if needed for certain filters
        // for sobel filter or edge detection filter we need grayscale image so converting those to grayscale image.
        if (input.channels() == 3 && (filterType == SOBEL_Y || filterType == EDGE_DETECTION)) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        // switch case statement to choose the filter type
        switch(filterType) {
            case GAUSSIAN: {
                // For Gaussian, use the filter's built-in method (modify if needed)
                result = gaussianFilter.apply(input, 5, 1.4);
                break;
            }
            // Detects VERTICAL edges.
            case SOBEL_X: {
                std::vector<std::vector<float>> sobelX = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
                };
                result = applyConvolutionVariant(grayInput, sobelX, variant);
                break;
            }
            // Detects HORIZONTAL edges.
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
                std::vector<std::vector<float>> sharpenKernel = {
                    { 0, -1,  0},
                    {-1,  5, -1},
                    { 0, -1,  0}
                };
                result = applyConvolutionVariant(input, sharpenKernel, variant);
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
    
    cv::Mat applyConvolutionVariant(const cv::Mat& input, const std::vector<std::vector<float>>& kernel,ConvolutionVariant variant) {
        cv::Mat result;
        
        switch(variant) {

            case STANDARD:
                result = ConvolutionEngineOMP::convolve2D(input, kernel);
                break;

            case BALANCED:
                result = ConvolutionEngineOMP::convolve2DBalanced(input, kernel);
                break;

            case CACHE_OPTIMIZED:
                result = ConvolutionEngineOMP::convolve2DCacheOptimized(input, kernel);
                break;

            case RAW_ARRAY: {
                // Convert to raw array format
                cv::Mat floatInput;
                input.convertTo(floatInput, CV_32F); // convert to float for accuracy.
                
                int width = input.cols;
                int height = input.rows;
                int kernelSize = kernel.size();
                
                // Flatten kernel as raw array is 1D and the kernal is 2D
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
                        inputArray[y * width + x] = floatInput.at<float>(y, x);
                    }
                }
                
                // Call raw array convolution
                ConvolutionEngineOMP::convolve2D(inputArray, outputArray, width, height,flatKernel.data(), kernelSize);
                
                // Convert back to Matrics
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
// =======================================================================================
// Utility functions which is used for making the enum values into a human readable form    
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
            case STANDARD: return "standard";
            case BALANCED: return "balanced";
            case CACHE_OPTIMIZED: return "cache_optimized";
            case RAW_ARRAY: return "raw_array";
            default: return "unknown";
        }
    }
    
    std::string getVariantDisplayName(ConvolutionVariant variant) {
        switch(variant) {
            case STANDARD: return "Standard";
            case BALANCED: return "Balanced";
            case CACHE_OPTIMIZED: return "Cache Optimized";
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
                    
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".bmp") {
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
// =============================================================================================

    // Function to process the image by taking image path, FilterType and which convolution
    void processImage(const std::string& imagePath, FilterType filterType, ConvolutionVariant variant) {

        std::string imageName = fs::path(imagePath).stem().string(); // get's the file path
        std::string filterName = getFilterName(filterType);// gets the name of the filter 
        std::string variantName = getVariantName(variant); // Gets which convolution varient we are going to do
        // these are used for displaying of the contents 
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
        
        std::vector<ConvolutionVariant> variants = {
            STANDARD, BALANCED, CACHE_OPTIMIZED, RAW_ARRAY
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
        reportFile << "  Total images processed: " << allResults.size() / 28 << "\n\n";
        
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
        std::string csvPath = outputFolder + "/reports/performance_data.csv";
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
    std::cout << "WITH ALL CONVOLUTION VARIANTS" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Default paths - modify these as needed
    std::string inputFolder = "C:\\Users\\DELL\\Downloads\\UCMerced_LandUse\\UCMerced_LandUse\\Images\\buildings";
    std::string outputFolder = "C:\\Users\\DELL\\hpc project\\Team-B5-HPC\\open_mp_implementation\\UCMerced_Output_Buildings";

    // Parse command line arguments
    if (argc > 1) inputFolder = argv[1];
    if (argc > 2) outputFolder = argv[2];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input folder: " << inputFolder << std::endl;
    std::cout << "  Output folder: " << outputFolder << std::endl;
    std::cout << "  System max threads according to OpenMP: " << omp_get_max_threads() << std::endl;

    // Check if input folder exists
    if (!fs::exists(inputFolder)) {
        std::cerr << "\nERROR: Input folder does not exist!" << std::endl;
        std::cerr << "Please check the path: " << inputFolder << std::endl;
        return 1;
    }

    // ========================================
    // THREAD SCALING TEST (FORCED ALL THREADS)
    // ========================================
    std::vector<int> threadCounts = {2, 4, 8, 16};
    std::cout << "\n========================================" << std::endl;
    std::cout << "THREAD SCALING PERFORMANCE TEST" << std::endl;
    std::cout << "========================================" << std::endl;

    for (int t : threadCounts) {
        std::cout << "\nRunning with " << t << " threads..." << std::endl;

        // Force the number of threads for OpenMP
        omp_set_num_threads(t);

        // Create a unique output folder for this thread count
        std::string threadOutputFolder = outputFolder + "_T" + std::to_string(t);
        if (!fs::exists(threadOutputFolder)) {
            fs::create_directories(threadOutputFolder);
        }

        // Create benchmark instance
        UCMercedBuildingsBenchmark benchmark(inputFolder, threadOutputFolder, t);

        // Process all images with all variants
        benchmark.processBatch();

        // Generate reports
        benchmark.generateReport();
        benchmark.generateCSVReport();

        std::cout << "Results saved in folder: " << threadOutputFolder << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "PROCESSING COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nVariants tested:" << std::endl;
    std::cout << "  1. Standard Convolution" << std::endl;
    std::cout << "  2. Balanced Load Convolution" << std::endl;
    std::cout << "  3. Cache-Optimized Convolution" << std::endl;
    std::cout << "  4. Raw Array Convolution" << std::endl;

    return 0;
}
