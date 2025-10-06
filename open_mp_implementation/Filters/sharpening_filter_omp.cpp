#include "sharpening_filter.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>

// Sharpening kernel (3x3)
const std::vector<std::vector<int>> sharpeningKernelOMP = {
    { 0, -1,  0 },
    {-1,  5, -1},
    { 0, -1,  0 }
};

// Apply sharpening to one channel with OpenMP
cv::Mat applySharpenChannelOMP(const cv::Mat& input) {
    int padding = 1;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    #pragma omp parallel for collapse(2)
    for (int i = padding; i < input.rows - padding; i++) {
        for (int j = padding; j < input.cols - padding; j++) {
            int sum = 0;
            for (int ki = -padding; ki <= padding; ki++) {
                for (int kj = -padding; kj <= padding; kj++) {
                    int pixel = input.at<uchar>(i + ki, j + kj);
                    sum += pixel * sharpeningKernelOMP[ki + padding][kj + padding];
                }
            }
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output;
}

// OpenMP Sharpening filter class
cv::Mat SharpeningFilterOMP::apply(const cv::Mat& input) {
    std::cout << "=== PARALLEL SHARPENING FILTER (OpenMP) ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat output;
    if (input.channels() == 1) {
        output = applySharpenChannelOMP(input);
    } else if (input.channels() == 3) {
        std::vector<cv::Mat> channels(3), outChannels(3);
        cv::split(input, channels);

        #pragma omp parallel for
        for (int c = 0; c < 3; c++)
            outChannels[c] = applySharpenChannelOMP(channels[c]);

        cv::merge(outChannels, output);
    } else {
        throw std::runtime_error("Unsupported number of channels.");
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel processing time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    return output;
}
