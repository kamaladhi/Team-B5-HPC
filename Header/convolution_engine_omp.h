// convolution_engine_omp.h
// This is used for defining the header ifndef - if not defined we need to define this header else this file will be skipped because header is already defined in another file.
//This will help me defining the class only once even if multiple files include this header.

#ifndef CONVOLUTION_ENGINE_OMP_H
#define CONVOLUTION_ENGINE_OMP_H

#include <opencv2/opencv.hpp> //include  the open cv library for image filtering
#include <vector>  // dynamic array type 
#include <omp.h> // open mp library for parallel implementation.

class ConvolutionEngineOMP {
public:
    // Raw array convolution with OpenMP
    static void convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize);
    // Representation of array: std::vector<float> similarly for 2d array std::vector<std:vector<float>>
    // OpenCV Mat convolution with OpenMP
    static cv::Mat convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);

    // Parallel convolution with load balancing (dynamic scheduling)
    static cv::Mat convolve2DBalanced(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);

    // Cache-optimized parallel convolution uses cache to store the reusable pixel value.
    static cv::Mat convolve2DCacheOptimized(const cv::Mat& input, const std::vector<std::vector<float>>& kernel);
};

#endif // CONVOLUTION_ENGINE_OMP_H
