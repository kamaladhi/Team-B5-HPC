#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

// Filters
#include "gaussian_filter.h"
#include "sobel_filter.h"
#include "sharpening_filter.h"

// Utility: measure execution time of a function
template <typename Func>
cv::Mat runAndMeasure(const std::string& name, bool isOMP,
                      Func func, const cv::Mat& input, const std::string& outFile) {
    std::cout << "=== " << name << " ===" << std::endl;
    if (isOMP) {
        std::cout << "Threads available: " << omp_get_max_threads() << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result = func(input);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << name << " processing time: " << duration << " ms" << std::endl;

    if (!outFile.empty()) {
        cv::imwrite(outFile, result);
        std::cout << "Results saved: " << outFile << std::endl;
    }
    std::cout << std::endl;

    return result;
}

int main() {
    std::string imagePath = "test2.jpg";   // adjust if needed
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (input.empty()) {
        std::cerr << "Error: Could not open image at " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Loading image: " << imagePath << std::endl;
    std::cout << "Image loaded: " << input.rows << "x" << input.cols 
              << " channels: " << input.channels() << std::endl << std::endl;

    GaussianFilter gFilter;
    SobelFilter sFilter;
    SharpeningFilter shFilter;

    // Gaussian
    runAndMeasure("SEQUENTIAL GAUSSIAN FILTERING", false,
                  [&](const cv::Mat& img){ return gFilter.applySequential(img, 5, 1.0); },
                  input, "results/gaussian_sequential.jpg");

    runAndMeasure("OPENMP GAUSSIAN FILTERING", true,
                  [&](const cv::Mat& img){ return gFilter.applyOpenMP(img, 5, 1.0); },
                  input, "results/gaussian_openmp.jpg");

    // Sobel
    runAndMeasure("SEQUENTIAL SOBEL FILTERING", false,
                  [&](const cv::Mat& img){ return sFilter.applySequential(img); },
                  input, "results/sobel_sequential.jpg");

    runAndMeasure("OPENMP SOBEL FILTERING", true,
                  [&](const cv::Mat& img){ return sFilter.applyOpenMP(img); },
                  input, "results/sobel_openmp.jpg");

    // Sharpening
    runAndMeasure("SEQUENTIAL SHARPENING FILTERING", false,
                  [&](const cv::Mat& img){ return shFilter.applySequential(img); },
                  input, "results/sharpen_sequential.jpg");

    runAndMeasure("OPENMP SHARPENING FILTERING", true,
                  [&](const cv::Mat& img){ return shFilter.applyOpenMP(img); },
                  input, "results/sharpen_openmp.jpg");

    std::cout << "All processing completed.\n";
    return 0;
}
