#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include <opencv2/opencv.hpp>

class ConvolutionEngine {
public:
    static void convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize);
    static cv::Mat convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);
};

#endif