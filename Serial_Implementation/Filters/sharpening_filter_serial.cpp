#include "sharpening_filter.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>

// Sharpening kernel (3x3)
const std::vector<std::vector<int>> sharpeningKernelSerial = {
    { 0, -1,  0 },
    {-1,  5, -1},
    { 0, -1,  0 }
};

// Apply sharpening to one channel (serial)
cv::Mat applySharpenChannelSerial(const cv::Mat& input) {
    int padding = 1;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    for (int i = padding; i < input.rows - padding; i++) {
        for (int j = padding; j < input.cols - padding; j++) {
            int sum = 0;
            for (int ki = -padding; ki <= padding; ki++) {
                for (int kj = -padding; kj <= padding; kj++) {
                    int pixel = input.at<uchar>(i + ki, j + kj);
                    sum += pixel * sharpeningKernelSerial[ki + padding][kj + padding];
                }
            }
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output;
}

cv::Mat SharpeningFilterSerial::apply(const cv::Mat& input) {
    std::cout << "=== SEQUENTIAL SHARPENING FILTER ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat output;
    if (input.channels() == 1) {
        output = applySharpenChannelSerial(input);
    } else if (input.channels() == 3) {
        std::vector<cv::Mat> channels(3), outChannels(3);
        cv::split(input, channels);
        for (int c = 0; c < 3; c++)
            outChannels[c] = applySharpenChannelSerial(channels[c]);
        cv::merge(outChannels, output);
    } else {
        throw std::runtime_error("Unsupported number of channels.");
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential processing time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    return output;
}
