#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "convolution_engine.h"
#include "performance_measure.h"

 // match your actual filename

#include "gaussian_filter.h"
#include "sobel_filter.h"
#include "sharpening_filter.h"
#include "laplacian_filter.h"

class SerialImageProcessor {
private:
    PerformanceMeasurement performance;
    
public:
    void processGaussianFilter(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== GAUSSIAN FILTER PROCESSING ===" << std::endl;
        
        performance.startTimer();
        GaussianFilterSerial gFilter;
        cv::Mat result = gFilter.apply(input, 5, 1.5);

        performance.stopTimer();
        
        performance.printResults("Gaussian Filter");
        PerformanceMeasurement::validateResult(result, input.cols, input.rows, input.channels());
        
        cv::imwrite(outputPath + "_gaussian.jpg", result);
        std::cout << "Gaussian filtered image saved to: " << outputPath + "_gaussian.jpg" << std::endl;
    }
    
    void processSobelFilter(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== SOBEL FILTER PROCESSING ===" << std::endl;
        
        performance.startTimer();
        SobelFilterSerial sFilter;
        cv::Mat result = sFilter.apply(input);

        performance.stopTimer();
        
        performance.printResults("Sobel Filter");
        PerformanceMeasurement::validateResult(result, input.cols, input.rows, input.channels());
        
        cv::imwrite(outputPath + "_sobel.jpg", result);
        std::cout << "Sobel filtered image saved to: " << outputPath + "_sobel.jpg" << std::endl;
    }
    
    void processSharpeningFilter(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== SHARPENING FILTER PROCESSING ===" << std::endl;
        
        performance.startTimer();
        SharpeningFilterSerial shFilter;
        cv::Mat result = shFilter.apply(input);

        performance.stopTimer();
        
        performance.printResults("Sharpening Filter");
        PerformanceMeasurement::validateResult(result, input.cols, input.rows, input.channels());
        
        cv::imwrite(outputPath + "_sharpened.jpg", result);
        std::cout << "Sharpened image saved to: " << outputPath + "_sharpened.jpg" << std::endl;
    }
    
    void processLaplacianFilter(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== LAPLACIAN FILTER PROCESSING ===" << std::endl;
        
        performance.startTimer();
        LaplacianFilterSerial lFilter;
        cv::Mat result = lFilter.apply(input);

        performance.stopTimer();
        
        performance.printResults("Laplacian Filter");
        PerformanceMeasurement::validateResult(result, input.cols, input.rows, input.channels());
        
        cv::imwrite(outputPath + "_laplacian.jpg", result);
        std::cout << "Laplacian filtered image saved to: " << outputPath + "_laplacian.jpg" << std::endl;
    }
    
    void processCustomConvolution(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== CUSTOM CONVOLUTION ENGINE ===" << std::endl;
        
        std::vector<std::vector<float>> edgeKernel = {
            {-1, -1, -1},
            {-1,  8, -1},
            {-1, -1, -1}
        };
        
        performance.startTimer();
        
        cv::Mat result;
        if (input.channels() == 1) {
            result = ConvolutionEngine::convolve2D(input, edgeKernel);
        } else {
            std::vector<cv::Mat> channels, outChannels;
            cv::split(input, channels);
            
            for (int c = 0; c < input.channels(); c++) {
                cv::Mat channelResult = ConvolutionEngine::convolve2D(channels[c], edgeKernel);
                outChannels.push_back(channelResult);
            }
            cv::merge(outChannels, result);
        }
        
        performance.stopTimer();
        
        cv::Mat finalResult;
        result.convertTo(finalResult, CV_8U);
        
        performance.printResults("Custom Convolution");
        PerformanceMeasurement::validateResult(finalResult, input.cols, input.rows, input.channels());
        
        cv::imwrite(outputPath + "_custom_convolution.jpg", finalResult);
        std::cout << "Custom convolution result saved to: " << outputPath + "_custom_convolution.jpg" << std::endl;
    }
    
    void processRawConvolution(const cv::Mat& input, const std::string& outputPath) {
        std::cout << "\n=== RAW CONVOLUTION FUNCTION ===" << std::endl;
        
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        float* inputData = new float[grayInput.rows * grayInput.cols];
        float* outputData = new float[grayInput.rows * grayInput.cols];
        
        for (int y = 0; y < grayInput.rows; y++) {
            for (int x = 0; x < grayInput.cols; x++) {
                inputData[y * grayInput.cols + x] = static_cast<float>(grayInput.at<uchar>(y, x));
            }
        }
        
        float blurKernel[9] = {
            0.111f, 0.111f, 0.111f,
            0.111f, 0.111f, 0.111f,
            0.111f, 0.111f, 0.111f
        };
        
        performance.startTimer();
        ConvolutionEngine::convolve2D(inputData, outputData, grayInput.cols, grayInput.rows, blurKernel, 3);
        performance.stopTimer();
        
        cv::Mat result = cv::Mat::zeros(grayInput.size(), CV_8U);
        for (int y = 0; y < grayInput.rows; y++) {
            for (int x = 0; x < grayInput.cols; x++) {
                result.at<uchar>(y, x) = cv::saturate_cast<uchar>(outputData[y * grayInput.cols + x]);
            }
        }
        
        performance.printResults("Raw Convolution");
        PerformanceMeasurement::validateResult(result, grayInput.cols, grayInput.rows, 1);
        
        cv::imwrite(outputPath + "_raw_convolution.jpg", result);
        std::cout << "Raw convolution result saved to: " << outputPath + "_raw_convolution.jpg" << std::endl;
        
        delete[] inputData;
        delete[] outputData;
    }
    
    void runComprehensiveBenchmark(const cv::Mat& input) {
        std::cout << "\n=== COMPREHENSIVE BENCHMARK ===" << std::endl;
        std::cout << "Image size: " << input.cols << "x" << input.rows << std::endl;
        std::cout << "Channels: " << input.channels() << std::endl;
        std::cout << "Memory usage before processing: " << PerformanceMeasurement::getMemoryUsage() << " KB" << std::endl;
        
        cv::Mat original = input.clone();
        
        processGaussianFilter(input, "output");
        processSobelFilter(input, "output");
        processSharpeningFilter(input, "output");
        processLaplacianFilter(input, "output");
        processCustomConvolution(input, "output");
        processRawConvolution(input, "output");
        
        std::cout << "\nMemory usage after processing: " << PerformanceMeasurement::getMemoryUsage() << " KB" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== SERIAL IMPLEMENTATION PHASE 2 ===" << std::endl;
    
    std::string imagePath = (argc > 1) ? argv[1] : "C:/Users/DELL/hpc project/Image-Filtering-/test2.jpg";

    
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "Creating synthetic test image..." << std::endl;
        image = cv::Mat::zeros(512, 512, CV_8UC3);
        cv::rectangle(image, cv::Point(100, 100), cv::Point(400, 400), cv::Scalar(255, 255, 255), -1);
        cv::circle(image, cv::Point(256, 256), 50, cv::Scalar(128, 128, 128), -1);
        for (int i = 0; i < 1000; i++) {
            int x = rand() % image.cols;
            int y = rand() % image.rows;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        }
        cv::imwrite("synthetic_test.jpg", image);
        std::cout << "Synthetic test image created and saved as synthetic_test.jpg" << std::endl;
    } else {
        std::cout << "Loaded image: " << imagePath << std::endl;
    }
    
    SerialImageProcessor processor;
    processor.runComprehensiveBenchmark(image);
    
    std::cout << "\n=== PROCESSING COMPLETE ===" << std::endl;
    std::cout << "All filtered images have been saved with respective suffixes." << std::endl;
    
    return 0;
}