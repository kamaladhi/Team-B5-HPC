#include "laplacian_filter.h"
#include <vector>
#include <stdexcept>
#include <omp.h>

// Laplacian kernel (3x3)
const std::vector<std::vector<int>> laplacianKernelOMP = {
    { 0, -1,  0 },
    {-1,  4, -1},
    { 0, -1,  0 }
};

// Apply Laplacian to one channel (parallel)
cv::Mat applyLaplacianChannelOMP(const cv::Mat& input) {
    int pad = 1;
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    #pragma omp parallel for collapse(2)
    for (int i = pad; i < input.rows - pad; i++) {
        for (int j = pad; j < input.cols - pad; j++) {
            int sum = 0;
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    int pixel = input.at<uchar>(i + ki, j + kj);
                    sum += pixel * laplacianKernelOMP[ki + pad][kj + pad];
                }
            }
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output;
}

cv::Mat LaplacianFilterOMP::apply(const cv::Mat& input) {
    if (input.channels() == 1) {
        return applyLaplacianChannelOMP(input);
    }

    std::vector<cv::Mat> channels, outChannels(3);
    cv::split(input, channels);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        outChannels[c] = applyLaplacianChannelOMP(channels[c]);
    }

    cv::Mat output;
    cv::merge(outChannels, output);
    return output;
}
