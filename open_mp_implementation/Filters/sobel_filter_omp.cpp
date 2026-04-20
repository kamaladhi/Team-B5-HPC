#include "sobel_filter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

// Sobel kernels (defined in header)
const std::vector<std::vector<int>> sobelX = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const std::vector<std::vector<int>> sobelY = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Apply Sobel filter to one channel (parallel)
cv::Mat applySobelChannelOMP(const cv::Mat& input) {
    int padding = 1;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = padding; i < input.rows - padding; i++) {
        for (int j = padding; j < input.cols - padding; j++) {
            int gx = 0, gy = 0;
            
            // Unrolled kernel application for better performance
            for (int ki = -padding; ki <= padding; ki++) {
                for (int kj = -padding; kj <= padding; kj++) {
                    int pixel = input.at<uchar>(i + ki, j + kj);
                    gx += pixel * sobelX[ki + padding][kj + padding];
                    gy += pixel * sobelY[ki + padding][kj + padding];
                }
            }
            
            int mag = static_cast<int>(std::sqrt(gx*gx + gy*gy));
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(mag);
        }
    }
    return output;
}

cv::Mat SobelFilterOMP::apply(const cv::Mat& input) {
    cv::Mat output;
    
    if (input.channels() == 1) {
        output = applySobelChannelOMP(input);
    } else if (input.channels() == 3) {
        std::vector<cv::Mat> channels(3), outChannels(3);
        cv::split(input, channels);

        // Process each channel in parallel
        #pragma omp parallel for
        for (int c = 0; c < 3; c++) {
            outChannels[c] = applySobelChannelOMP(channels[c]);
        }

        cv::merge(outChannels, output);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    
    return output;
}

cv::Mat SobelFilterOMP::applyOptimized(const cv::Mat& input) {
    cv::Mat output;
    
    if (input.channels() == 1) {
        // Cache-optimized version with tiling
        output = cv::Mat::zeros(input.size(), input.type());
        const int TILE_SIZE = 64;
        int padding = 1;
        
        #pragma omp parallel for collapse(2)
        for (int ty = padding; ty < input.rows - padding; ty += TILE_SIZE) {
            for (int tx = padding; tx < input.cols - padding; tx += TILE_SIZE) {
                int endY = std::min(ty + TILE_SIZE, input.rows - padding);
                int endX = std::min(tx + TILE_SIZE, input.cols - padding);
                
                // Process tile
                for (int i = ty; i < endY; i++) {
                    for (int j = tx; j < endX; j++) {
                        int gx = 0, gy = 0;
                        
                        for (int ki = -padding; ki <= padding; ki++) {
                            for (int kj = -padding; kj <= padding; kj++) {
                                int pixel = input.at<uchar>(i + ki, j + kj);
                                gx += pixel * sobelX[ki + padding][kj + padding];
                                gy += pixel * sobelY[ki + padding][kj + padding];
                            }
                        }
                        
                        int mag = static_cast<int>(std::sqrt(gx*gx + gy*gy));
                        output.at<uchar>(i, j) = cv::saturate_cast<uchar>(mag);
                    }
                }
            }
        }
    } else {
        output = apply(input); // Fall back to standard implementation
    }
    
    return output;
}