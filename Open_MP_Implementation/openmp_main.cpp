#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

#include "convolution_engine_omp.h"
#include "performance_measure_omp.h"
#include "gaussian_filter.h"
#include "sobel_filter.h"
#include "sharpening_filter.h"
#include "laplacian_filter.h"

class OpenMPImageProcessor {
private:
PerformanceMeasurementOMP performance;

public:
void processGaussianFilter(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== GAUSSIAN FILTER PROCESSING (OpenMP - " << numThreads << " threads) ===" << std::endl;

omp_set_num_threads(numThreads);
performance.startTimer();

GaussianFilterOMP gFilter;
cv::Mat result = gFilter.apply(input, 5, 1.5);

performance.stopTimer();

performance.printResults("Gaussian Filter (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(result, input.cols, input.rows, input.channels());

std::string filename = outputPath + "_gaussian_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, result);
std::cout << "Gaussian filtered image saved to: " << filename << std::endl;
}

void processSobelFilter(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== SOBEL FILTER PROCESSING (OpenMP - " << numThreads << " threads) ===" << std::endl;

omp_set_num_threads(numThreads);
performance.startTimer();

SobelFilterOMP sFilter;
cv::Mat result = sFilter.apply(input);

performance.stopTimer();

performance.printResults("Sobel Filter (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(result, input.cols, input.rows, input.channels());

std::string filename = outputPath + "_sobel_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, result);
std::cout << "Sobel filtered image saved to: " << filename << std::endl;
}

void processSharpeningFilter(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== SHARPENING FILTER PROCESSING (OpenMP - " << numThreads << " threads) ===" << std::endl;

omp_set_num_threads(numThreads);
performance.startTimer();

SharpeningFilterOMP shFilter;
cv::Mat result = shFilter.apply(input);

performance.stopTimer();

performance.printResults("Sharpening Filter (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(result, input.cols, input.rows, input.channels());

std::string filename = outputPath + "_sharpened_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, result);
std::cout << "Sharpened image saved to: " << filename << std::endl;
}

void processLaplacianFilter(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== LAPLACIAN FILTER PROCESSING (OpenMP - " << numThreads << " threads) ===" << std::endl;

omp_set_num_threads(numThreads);
performance.startTimer();

LaplacianFilterOMP lFilter;
cv::Mat result = lFilter.apply(input);

performance.stopTimer();

performance.printResults("Laplacian Filter (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(result, input.cols, input.rows, input.channels());

std::string filename = outputPath + "_laplacian_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, result);
std::cout << "Laplacian filtered image saved to: " << filename << std::endl;
}

void processCustomConvolution(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== CUSTOM CONVOLUTION ENGINE (OpenMP - " << numThreads << " threads) ===" << std::endl;

std::vector<std::vector<float>> edgeKernel = {
{-1, -1, -1},
{-1,  8, -1},
{-1, -1, -1}
};

omp_set_num_threads(numThreads);
performance.startTimer();

cv::Mat result;
if (input.channels() == 1) {
result = ConvolutionEngineOMP::convolve2D(input, edgeKernel);
} else {
std::vector<cv::Mat> channels, outChannels;
cv::split(input, channels);
outChannels.resize(input.channels());

#pragma omp parallel for
for (int c = 0; c < input.channels(); c++) {
outChannels[c] = ConvolutionEngineOMP::convolve2D(channels[c], edgeKernel);
}
cv::merge(outChannels, result);
}

performance.stopTimer();

cv::Mat finalResult;
result.convertTo(finalResult, CV_8U);

performance.printResults("Custom Convolution (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(finalResult, input.cols, input.rows, input.channels());

std::string filename = outputPath + "_custom_convolution_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, finalResult);
std::cout << "Custom convolution result saved to: " << filename << std::endl;
}

void processRawConvolution(const cv::Mat& input, const std::string& outputPath, int numThreads) {
std::cout << "\n=== RAW CONVOLUTION FUNCTION (OpenMP - " << numThreads << " threads) ===" << std::endl;

cv::Mat grayInput;
if (input.channels() == 3) {
cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
} else {
grayInput = input.clone();
}

float* inputData = new float[grayInput.rows * grayInput.cols];
float* outputData = new float[grayInput.rows * grayInput.cols];

// Copy input data
#pragma omp parallel for collapse(2)
for (int y = 0; y < grayInput.rows; y++) {
for (int x = 0; x < grayInput.cols; x++) {
inputData[y * grayInput.cols + x] = static_cast<float>(grayInput.at<uchar>(y, x));
}
}

float blurKernel[9] = {
0.111f, 0.111f, 0.111f,
0.111f, 0.111f, 0.111f,
0.111f, 0.111f, 0.111f
};

omp_set_num_threads(numThreads);
performance.startTimer();

ConvolutionEngineOMP::convolve2D(inputData, outputData, grayInput.cols, grayInput.rows, blurKernel, 3);

performance.stopTimer();

cv::Mat result = cv::Mat::zeros(grayInput.size(), CV_8U);
#pragma omp parallel for collapse(2)
for (int y = 0; y < grayInput.rows; y++) {
for (int x = 0; x < grayInput.cols; x++) {
result.at<uchar>(y, x) = cv::saturate_cast<uchar>(outputData[y * grayInput.cols + x]);
}
}

performance.printResults("Raw Convolution (OpenMP)", numThreads);
PerformanceMeasurementOMP::validateResult(result, grayInput.cols, grayInput.rows, 1);

std::string filename = outputPath + "_raw_convolution_omp_" + std::to_string(numThreads) + "t.jpg";
cv::imwrite(filename, result);
std::cout << "Raw convolution result saved to: " << filename << std::endl;

delete[] inputData;
delete[] outputData;
}

void runThreadScalabilityAnalysis(const cv::Mat& input) {
std::cout << "\n=== THREAD SCALABILITY ANALYSIS ===" << std::endl;
std::cout << "Maximum threads available: " << omp_get_max_threads() << std::endl;

std::vector<int> threadCounts = {1, 2, 4, 8, 12, 16};
std::vector<double> executionTimes;

// Test Gaussian filter with different thread counts
for (int threads : threadCounts) {
if (threads <= omp_get_max_threads()) {
std::cout << "\nTesting with " << threads << " threads..." << std::endl;

omp_set_num_threads(threads);
performance.startTimer();

GaussianFilterOMP gFilter;
cv::Mat result = gFilter.apply(input, 5, 1.5);

performance.stopTimer();

double execTime = performance.getElapsedMilliseconds();
executionTimes.push_back(execTime);

std::cout << "Threads: " << threads << ", Time: " << execTime << " ms" << std::endl;
}
}

// Calculate speedup
std::cout << "\n=== SPEEDUP ANALYSIS ===" << std::endl;
if (!executionTimes.empty()) {
double serialTime = executionTimes[0]; // 1 thread time
std::cout << std::fixed << std::setprecision(2);
std::cout << "Threads\tTime(ms)\tSpeedup\tEfficiency" << std::endl;
std::cout << "-------\t--------\t-------\t----------" << std::endl;

for (size_t i = 0; i < executionTimes.size() && i < threadCounts.size(); i++) {
double speedup = serialTime / executionTimes[i];
double efficiency = speedup / threadCounts[i];
std::cout << threadCounts[i] << "\t" << executionTimes[i] << "\t\t" 
 << speedup << "\t" << efficiency << std::endl;
}
}
}

void runComprehensiveBenchmark(const cv::Mat& input, int numThreads) {
std::cout << "\n=== COMPREHENSIVE OPENMP BENCHMARK ===" << std::endl;
std::cout << "Image size: " << input.cols << "x" << input.rows << std::endl;
std::cout << "Channels: " << input.channels() << std::endl;
std::cout << "Number of threads: " << numThreads << std::endl;
std::cout << "Memory usage before processing: " << PerformanceMeasurementOMP::getMemoryUsage() << " KB" << std::endl;

processGaussianFilter(input, "output", numThreads);
processSobelFilter(input, "output", numThreads);
processSharpeningFilter(input, "output", numThreads);
processLaplacianFilter(input, "output", numThreads);
processCustomConvolution(input, "output", numThreads);
processRawConvolution(input, "output", numThreads);

std::cout << "\nMemory usage after processing: " << PerformanceMeasurementOMP::getMemoryUsage() << " KB" << std::endl;
}

double getLastExecutionTime() {
return performance.getElapsedMilliseconds();
}
};

int main(int argc, char* argv[]) {
std::cout << "=== OPENMP IMPLEMENTATION PHASE 3 ===" << std::endl;

std::string imagePath = (argc > 1) ? argv[1] : "C:/Users/DELL/hpc project/Image-Filtering-/test2.jpg";
int numThreads = (argc > 2) ? std::stoi(argv[2]) : omp_get_max_threads();

std::cout << "Maximum threads available: " << omp_get_max_threads() << std::endl;
std::cout << "Using threads: " << numThreads << std::endl;

cv::Mat image = cv::imread(imagePath);
if (image.empty()) {
std::cout << "Creating synthetic test image..." << std::endl;
image = cv::Mat::zeros(512, 512, CV_8UC3);
cv::rectangle(image, cv::Point(100, 100), cv::Point(400, 400), cv::Scalar(255, 255, 255), -1);
cv::circle(image, cv::Point(256, 256), 50, cv::Scalar(128, 128, 128), -1);
for (int i = 0; i < 1000; i++) {
int x = rand() % image.cols;
int y = rand() % image.rows;
image.at<cv::Vec3b>(y, x) = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
}
cv::imwrite("synthetic_test.jpg", image);
std::cout << "Synthetic test image created and saved as synthetic_test.jpg" << std::endl;
} else {
std::cout << "Loaded image: " << imagePath << std::endl;
}

OpenMPImageProcessor processor;

// Run comprehensive benchmark
processor.runComprehensiveBenchmark(image, numThreads);

// Run thread scalability analysis
processor.runThreadScalabilityAnalysis(image);

std::cout << "\n=== PROCESSING COMPLETE ===" << std::endl;
std::cout << "All filtered images have been saved with respective suffixes." << std::endl;

return 0;
}