#include "convolution_engine.h"
#include <cstring>

void ConvolutionEngine::convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize) {
    int halfKernel = kernelSize / 2;
    
    memset(output, 0, width * height * sizeof(float));
    
    for (int y = halfKernel; y < height - halfKernel; y++) {
        for (int x = halfKernel; x < width - halfKernel; x++) {
            float sum = 0.0f;
            
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int inputY = y + ky;
                    int inputX = x + kx;
                    int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
                    int inputIdx = inputY * width + inputX;
                    
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

cv::Mat ConvolutionEngine::convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
    int kernelSize = kernel.size();
    int halfKernel = kernelSize / 2;
    
    cv::Mat floatInput;
    input.convertTo(floatInput, CV_32F);
    
    for (int y = halfKernel; y < input.rows - halfKernel; y++) {
        for (int x = halfKernel; x < input.cols - halfKernel; x++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int inputY = y + ky - halfKernel;
                    int inputX = x + kx - halfKernel;
                    
                    sum += floatInput.at<float>(inputY, inputX) * kernel[ky][kx];
                }
            }
            
            output.at<float>(y, x) = sum;
        }
    }
    
    return output;
}