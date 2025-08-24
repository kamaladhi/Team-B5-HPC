// convolution_engine_omp.cpp
#include "convolution_engine_omp.h"
#include <cstring>
#include <algorithm>


// Performing in array
void ConvolutionEngineOMP::convolve2D(float* input, float* output, int width, int height, float* kernel, int kernelSize) {
    int halfKernel = kernelSize / 2; // finding the radius of the kernal 

    // Initialize output array in parallel
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        output[i] = 0.0f;
        // Each and every time a 1d array with size of weight*height is created and values of outputs are being stored when going to next row the values in output array is flushed out and fresh 0 array will be passed .
        // This is because if the output contains garbage value that will affect the convolution .
        // Also multiple thread uses this pragma omp parallel for and each thread will write its own output so set to 0 initially.
    }

    // Parallel convolution with collapse directive for better load balancing
    #pragma omp parallel for collapse(2) schedule(dynamic)
    // loop starts by avoiding the border pixels this will prevent accessing  memory outside the input array.
    // computing over the image 
    for (int y = halfKernel; y < height - halfKernel; y++) { 
        for (int x = halfKernel; x < width - halfKernel; x++) {
            // Here there are two loops with y and x now this parallel for need to treat these as a single loop for load balancing.
            // shedule(dynamic) - shedules a dynamic work where the chunks are taken dynamically to avoid some threads being idle if one part of the image is heavier to compute.
            
            float sum = 0.0f;
            // Loops over the kernel
            // Core convolution processes taking place here.
            // Unroll kernel loops for better performance
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int inputY = y + ky;
                int inputX = x + kx;
                int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel); // converting it into 1D vector 
                int inputIdx = inputY * width + inputX;

                sum += input[inputIdx] * kernel[kernelIdx];
                }
            }

            output[y * width + x] = sum;
        }
    }
}

// performing in opencv:
cv::Mat ConvolutionEngineOMP::convolve2D(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F); // This line helps in creating a output matrix with the size same as input. // cv::Mat::zeros static function of cv::Mat.
    int kernelSize = kernel.size(); // getting the  kernal size from the parameter 
    int halfKernel = kernelSize / 2; // calculating the radius of the kernal.

    cv::Mat floatInput;
    input.convertTo(floatInput, CV_32F);  // cv_32F - means converting it into 32 bit float value

    // Parallel convolution with OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
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

cv::Mat ConvolutionEngineOMP::convolve2DBalanced(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
    int kernelSize = kernel.size();
    int halfKernel = kernelSize / 2;

    cv::Mat floatInput;
    input.convertTo(floatInput, CV_32F);

    int effectiveRows = input.rows - 2 * halfKernel; // removing the borders of the image in top and bottom
    int effectiveCols = input.cols - 2 * halfKernel; // removing the borders of the image in left and right
    int totalPixels = effectiveRows * effectiveCols; // this is the pixel value where kernal will slide over (or kernel center) or total number of pixel where convolution is applied 

    // Dynamic scheduling with chunk size optimization
    int chunkSize = std::max(1, totalPixels / (omp_get_max_threads() * 4));  // omp_get_max_threads() returns the maximum number of threads the open mp will use.
    // smaller chunks less idle time
    // instead of giving the full share of the thread we give 1/4th of it so multiplying with 4. This allows threads to pick new chunks dynamically when they finish early.

    #pragma omp parallel for schedule(dynamic, chunkSize)
    for (int idx = 0; idx < totalPixels; idx++) {
        int y = (idx / effectiveCols) + halfKernel;
        int x = (idx % effectiveCols) + halfKernel;

        float sum = 0.0f;

        for (int ky = 0; ky < kernelSize; ky++) { // we treat everything as a 1D array 
            for (int kx = 0; kx < kernelSize; kx++) {
                int inputY = y + ky - halfKernel;// then convert them into a 2D array
                int inputX = x + kx - halfKernel;

                sum += floatInput.at<float>(inputY, inputX) * kernel[ky][kx]; // does the convolution
            }
        }

        output.at<float>(y, x) = sum;
    }

    return output;
}

cv::Mat ConvolutionEngineOMP::convolve2DCacheOptimized(const cv::Mat& input, const std::vector<std::vector<float>>& kernel) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
    int kernelSize = kernel.size();
    int halfKernel = kernelSize / 2;

    cv::Mat floatInput;
    input.convertTo(floatInput, CV_32F);

    // Cache-friendly tiling
    const int TILE_SIZE = 64; // Adjust based on cache size
    // Image will be divided into chunks 
    #pragma omp parallel for collapse(2) schedule(static)
    // this two for loop iterates over the tiles and not by pixels 
    for (int ty = halfKernel; ty < input.rows - halfKernel; ty += TILE_SIZE) {
        for (int tx = halfKernel; tx < input.cols - halfKernel; tx += TILE_SIZE) {
            int endY = std::min(ty + TILE_SIZE, input.rows - halfKernel);
            int endX = std::min(tx + TILE_SIZE, input.cols - halfKernel);

            // Process tile - each tile inner loop  is iterated over each pixel.
            for (int y = ty; y < endY; y++) {
                for (int x = tx; x < endX; x++) {
                    float sum = 0.0f;
                    // applying kernal
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
        }
    }

    return output;
