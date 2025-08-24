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

    // Parallel convolution with load balancing (dynamic scheduling)
    static cv::Mat convolve2DBalanced(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);

    // Cache-optimized parallel convolution
    static cv::Mat convolve2DCacheOptimized(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);
};

#endif // CONVOLUTION_ENGINE_OMP_H
