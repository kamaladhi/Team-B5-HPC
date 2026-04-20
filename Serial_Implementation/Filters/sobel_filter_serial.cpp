#include "sobel_filter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

// Sobel kernels
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

// Apply Sobel filter to one channel (serial)
cv::Mat applySobelChannelSerial(const cv::Mat& input) {
    int padding = 1;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    for (int i = padding; i < input.rows - padding; i++) {
        for (int j = padding; j < input.cols - padding; j++) {
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
    return output;
}

cv::Mat SobelFilterSerial::apply(const cv::Mat& input) {
    cv::Mat output;
    if (input.channels() == 1) {
        output = applySobelChannelSerial(input);
    } else if (input.channels() == 3) {
        std::vector<cv::Mat> channels(3), outChannels(3);
        cv::split(input, channels);
        for (int c = 0; c < 3; c++) {
            outChannels[c] = applySobelChannelSerial(channels[c]);
        }
        cv::merge(outChannels, output);
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
    return output;
}
