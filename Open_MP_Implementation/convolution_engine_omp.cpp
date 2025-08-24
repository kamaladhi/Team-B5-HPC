// convolution_engine_omp.h
#ifndef CONVOLUTION_ENGINE_OMP_H
#define CONVOLUTION_ENGINE_OMP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>

class ConvolutionEngineOMP {
public:
// Raw array convolution with OpenMP
static void convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize);

// OpenCV Mat convolution with OpenMP
static cv::Mat convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);

// Advanced convolution with load balancing
static cv::Mat convolve2DBalanced(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);

// Cache-optimized convolution
static cv::Mat convolve2DCacheOptimized(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);
};

#endif // CONVOLUTION_ENGINE_OMP_H

// convolution_engine_omp.cpp
#include "convolution_engine_omp.h"
#include <cstring>
#include <algorithm>

void ConvolutionEngineOMP::convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize) {
int halfKernel = kernelSize / 2;

// Initialize output array in parallel
#pragma omp parallel for
for (int i = 0; i < width * height; i++) {
output[i] = 0.0f;
}

// Parallel convolution with collapse directive for better load balancing
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int y = halfKernel; y < height - halfKernel; y++) {
for (int x = halfKernel; x < width - halfKernel; x++) {
float sum = 0.0f;

// Unroll kernel loops for better performance
for (int ky = -halfKernel; ky <= halfKernel; ky++) {
for (int kx = -halfKernel; kx <= halfKernel; kx++) {
int inputY = y + ky;
int inputX = x + kx;
int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
int inputIdx = inputY * width + inputX;

sum += input[inputIdx] * kernel[kernelIdx];
}
}

output[y * width + x] = sum;
}
}
}

cv::Mat ConvolutionEngineOMP::convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
int kernelSize = kernel.size();
int halfKernel = kernelSize / 2;

cv::Mat floatInput;
input.convertTo(floatInput, CV_32F);

// Parallel convolution with OpenMP
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int y = halfKernel; y < input.rows - halfKernel; y++) {
for (int x = halfKernel; x < input.cols - halfKernel; x++) {
float sum = 0.0f;

for (int ky = 0; ky < kernelSize; ky++) {
for (int kx = 0; kx < kernelSize; kx++) {
int inputY = y + ky - halfKernel;
int inputX = x + kx - halfKernel;

sum += floatInput.at<float>(inputY, inputX) * kernel[ky][kx];
}
}

output.at<float>(y, x) = sum;
}
}

return output;
}

cv::Mat ConvolutionEngineOMP::convolve2DBalanced(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
int kernelSize = kernel.size();
int halfKernel = kernelSize / 2;

cv::Mat floatInput;
input.convertTo(floatInput, CV_32F);

int effectiveRows = input.rows - 2 * halfKernel;
int effectiveCols = input.cols - 2 * halfKernel;
int totalPixels = effectiveRows * effectiveCols;

// Dynamic scheduling with chunk size optimization
int chunkSize = std::max(1, totalPixels / (omp_get_max_threads() * 4));

#pragma omp parallel for schedule(dynamic, chunkSize)
for (int idx = 0; idx < totalPixels; idx++) {
int y = (idx / effectiveCols) + halfKernel;
int x = (idx % effectiveCols) + halfKernel;

float sum = 0.0f;

for (int ky = 0; ky < kernelSize; ky++) {
for (int kx = 0; kx < kernelSize; kx++) {
int inputY = y + ky - halfKernel;
int inputX = x + kx - halfKernel;

sum += floatInput.at<float>(inputY, inputX) * kernel[ky][kx];
}
}

output.at<float>(y, x) = sum;
}

return output;
}

cv::Mat ConvolutionEngineOMP::convolve2DCacheOptimized(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
int kernelSize = kernel.size();
int halfKernel = kernelSize / 2;

cv::Mat floatInput;
input.convertTo(floatInput, CV_32F);

// Cache-friendly tiling
const int TILE_SIZE = 64; // Adjust based on cache size

#pragma omp parallel for collapse(2) schedule(static)
for (int ty = halfKernel; ty < input.rows - halfKernel; ty += TILE_SIZE) {
for (int tx = halfKernel; tx < input.cols - halfKernel; tx += TILE_SIZE) {
int endY = std::min(ty + TILE_SIZE, input.rows - halfKernel);
int endX = std::min(tx + TILE_SIZE, input.cols - halfKernel);

// Process tile
for (int y = ty; y < endY; y++) {
for (int x = tx; x < endX; x++) {
float sum = 0.0f;

for (int ky = 0; ky < kernelSize; ky++) {
for (int kx = 0; kx < kernelSize; kx++) {
int inputY = y + ky - halfKernel;
int inputX = x + kx - halfKernel;

sum += floatInput.at<float>(inputY, inputX) * kernel[ky][kx];
}
}

output.at<float>(y, x) = sum;
}
}
}
}

return output;
}