
#include "gaussian_filter.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

cv::Mat GaussianFilterOMP::apply(const cv::Mat& input, int kernelSize, double sigma) {
    // Handle multi-channel images
    if (input.channels() == 3) {
        return applyMultiChannel(input, kernelSize, sigma);
    }
    
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // Create Gaussian kernel
    int k = kernelSize / 2;
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;

    // Generate kernel in parallel
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double value = exp(-(i*i + j*j) / (2 * sigma * sigma));
            kernel[i + k][j + k] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    // Apply convolution with optimized scheduling
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = k; y < input.rows - k; y++) {
        for (int x = k; x < input.cols - k; x++) {
            double val = 0.0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    val += input.at<uchar>(y + i, x + j) * kernel[i + k][j + k];
                }
            }
            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
        }
    }
    return output;
}

cv::Mat GaussianFilterOMP::applyMultiChannel(const cv::Mat& input, int kernelSize, double sigma) {
    std::vector<cv::Mat> channels, outChannels(3);
    cv::split(input, channels);
    
    // Process each channel in parallel
    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        outChannels[c] = apply(channels[c], kernelSize, sigma);
    }
    
    cv::Mat output;
    cv::merge(outChannels, output);
    return output;
}