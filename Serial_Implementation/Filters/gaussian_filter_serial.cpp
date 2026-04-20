#include "gaussian_filter.h"
#include <cmath>

cv::Mat GaussianFilterSerial::apply(const cv::Mat& input, int kernelSize, double sigma) {
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // Create Gaussian kernel
    int k = kernelSize / 2;
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;

    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            kernel[i + k][j + k] = exp(-(i*i + j*j) / (2 * sigma * sigma));
            sum += kernel[i + k][j + k];
        }
    }
    for (int i = 0; i < kernelSize; i++)
        for (int j = 0; j < kernelSize; j++)
            kernel[i][j] /= sum;

    // Convolution
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
