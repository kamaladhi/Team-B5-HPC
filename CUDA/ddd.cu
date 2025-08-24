#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <random> 

#define BLOCK_DIM 16

__device__ int clamp(int v, int low, int high) {
    return v < low ? low : (v > high ? high : v);
}

__global__ void gaussianKernel_sm(
    const unsigned char* input, unsigned char* output,
    int width, int height, size_t pitch, 
    const float* filter, int ksize, int channels)
{
    extern __shared__ unsigned char s_tile[];
    int halo = ksize / 2;
    int tile_dim = BLOCK_DIM + 2 * halo;
    
    int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_start_x = blockIdx.x * BLOCK_DIM - halo;
    int block_start_y = blockIdx.y * BLOCK_DIM - halo;

    for (int y_offset = ty; y_offset < tile_dim; y_offset += BLOCK_DIM) {
        for (int x_offset = tx; x_offset < tile_dim; x_offset += BLOCK_DIM) {
            int current_global_x = block_start_x + x_offset;
            int current_global_y = block_start_y + y_offset;

            current_global_x = clamp(current_global_x, 0, width - 1);
            current_global_y = clamp(current_global_y, 0, height - 1);
            
            for (int c = 0; c < channels; c++) {
                s_tile[(y_offset * tile_dim + x_offset) * channels + c] = 
                    input[current_global_y * pitch + current_global_x * channels + c];
            }
        }
    }
    __syncthreads();

    if (gx >= width || gy >= height) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f, norm = 0.0f;
        for (int fy = -halo; fy <= halo; fy++) {
            for (int fx = -halo; fx <= halo; fx++) {
                float w = filter[(fy + halo) * ksize + (fx + halo)];
                sum += w * s_tile[((ty + halo + fy) * tile_dim + (tx + halo + fx)) * channels + c];
                norm += w;
            }
        }
        output[gy * pitch + gx * channels + c] = (unsigned char)(sum / norm); 
    }
}

void cudaGaussian_advanced(const cv::Mat& src, cv::Mat& dst, int ksize, double sigma) {
    int width = src.cols, height = src.rows, channels = src.channels();
    dst.create(src.size(), src.type());

    std::vector<float> h_filter(ksize * ksize);
    int half = ksize / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float v = expf(-(x * x + y * y) / (2 * sigma * sigma));
            h_filter[(y + half) * ksize + (x + half)] = v;
            sum += v;
        }
    }
    for (auto& v : h_filter) v /= sum;

    unsigned char* d_in = nullptr, * d_out = nullptr;
    float* d_filter = nullptr;
    size_t bytes = width * height * channels * sizeof(unsigned char);
    
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_filter, ksize * ksize * sizeof(float));

    cudaMemcpy(d_in, src.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    int halo = ksize / 2;
    size_t sm_size = (BLOCK_DIM + 2 * halo) * (BLOCK_DIM + 2 * halo) * channels * sizeof(unsigned char);
    
    gaussianKernel_sm<<<grid, block, sm_size>>>(d_in, d_out, width, height, src.step[0], d_filter, ksize, channels); 
    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_filter);
}

__global__ void bilateralKernel_sm(
    const unsigned char* input, unsigned char* output,
    int width, int height, size_t pitch, 
    int diameter, float sigmaSpace, float sigmaColor, int channels)
{
    extern __shared__ unsigned char s_tile[];
    int radius = diameter / 2;
    int tile_dim = BLOCK_DIM + 2 * radius;
    
    int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_start_x = blockIdx.x * BLOCK_DIM - radius;
    int block_start_y = blockIdx.y * BLOCK_DIM - radius;

    for (int y_offset = ty; y_offset < tile_dim; y_offset += BLOCK_DIM) {
        for (int x_offset = tx; x_offset < tile_dim; x_offset += BLOCK_DIM) {
            int current_global_x = block_start_x + x_offset;
            int current_global_y = block_start_y + y_offset;

            current_global_x = clamp(current_global_x, 0, width - 1);
            current_global_y = clamp(current_global_y, 0, height - 1);
            
            for (int c = 0; c < channels; c++) {
                s_tile[(y_offset * tile_dim + x_offset) * channels + c] = 
                    input[current_global_y * pitch + current_global_x * channels + c];
            }
        }
    }
    __syncthreads();

    if (gx >= width || gy >= height) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f, norm = 0.0f;
        float centerVal = s_tile[((ty + radius) * tile_dim + (tx + radius)) * channels + c];

        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                float neighborVal = s_tile[((ty + radius + fy) * tile_dim + (tx + radius + fx)) * channels + c];
                float gs = expf(-(fx * fx + fy * fy) / (2 * sigmaSpace * sigmaSpace));
                float gc = expf(-((neighborVal - centerVal) * (neighborVal - centerVal)) / (2 * sigmaColor * sigmaColor));
                float w = gs * gc;
                sum += w * neighborVal;
                norm += w;
            }
        }
        output[gy * pitch + gx * channels + c] = (unsigned char)(sum / norm); 
    }
}
void cudaBilateral_advanced(const cv::Mat& src, cv::Mat& dst, int diameter, double sigmaSpace, double sigmaColor) {
    int width = src.cols, height = src.rows, channels = src.channels();
    dst.create(src.size(), src.type());

    unsigned char* d_in = nullptr, * d_out = nullptr;
    size_t bytes = width * height * channels * sizeof(unsigned char);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, src.data, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    int radius = diameter / 2;
    size_t sm_size = (BLOCK_DIM + 2 * radius) * (BLOCK_DIM + 2 * radius) * channels * sizeof(unsigned char);
    
    bilateralKernel_sm<<<grid, block, sm_size>>>(d_in, d_out, width, height, src.step[0], diameter, sigmaSpace, sigmaColor, channels); // FIXED: Call kernel, not self
    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out);
}
__global__ void nlmKernel_sm(
    const unsigned char* input, unsigned char* output,
    int width, int height, size_t pitch, 
    int searchRadius, int patchRadius, float h2, int channels)
{
    extern __shared__ unsigned char s_tile[];
    int halo = patchRadius + searchRadius; 
    int tile_dim = BLOCK_DIM + 2 * halo;
    
    int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_start_x = blockIdx.x * BLOCK_DIM - halo;
    int block_start_y = blockIdx.y * BLOCK_DIM - halo;

    for (int y_offset = ty; y_offset < tile_dim; y_offset += BLOCK_DIM) {
        for (int x_offset = tx; x_offset < tile_dim; x_offset += BLOCK_DIM) {
            int current_global_x = block_start_x + x_offset;
            int current_global_y = block_start_y + y_offset;

            current_global_x = clamp(current_global_x, 0, width - 1);
            current_global_y = clamp(current_global_y, 0, height - 1);
            
            for (int c = 0; c < channels; c++) {
                s_tile[(y_offset * tile_dim + x_offset) * channels + c] = 
                    input[current_global_y * pitch + current_global_x * channels + c];
            }
        }
    }
    __syncthreads();
    
    if (gx >= width || gy >= height) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f, norm = 0.0f;
        
        for (int sy = -searchRadius; sy <= searchRadius; sy++) {
            for (int sx = -searchRadius; sx <= searchRadius; sx++) {
                float dist2 = 0.0f;
                
                for (int py = -patchRadius; py <= patchRadius; py++) {
                    for (int px = -patchRadius; px <= patchRadius; px++) {
                        int current_patch_x = tx + halo + px;
                        int current_patch_y = ty + halo + py;
                        
                        int neighbor_patch_x = tx + halo + sx + px;
                        int neighbor_patch_y = ty + halo + sy + py;
                        
                        float diff = s_tile[(current_patch_y * tile_dim + current_patch_x) * channels + c] -
                                     s_tile[(neighbor_patch_y * tile_dim + neighbor_patch_x) * channels + c];
                        dist2 += diff * diff;
                    }
                }
                
                float w = expf(-dist2 / h2);
                int neighbor_x_sm = tx + halo + sx;
                int neighbor_y_sm = ty + halo + sy;
                sum += w * s_tile[(neighbor_y_sm * tile_dim + neighbor_x_sm) * channels + c];
                norm += w;
            }
        }
        output[gy * pitch + gx * channels + c] = (unsigned char)(sum / norm); 
    }
}

void cudaFastNLM_advanced(const cv::Mat& src, cv::Mat& dst, int searchRadius, int patchRadius, double h) {
    int width = src.cols, height = src.rows, channels = src.channels();
    dst.create(src.size(), src.type());

    unsigned char* d_in = nullptr, * d_out = nullptr;
    size_t bytes = width * height * channels * sizeof(unsigned char);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, src.data, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    int halo = searchRadius + patchRadius;
    size_t sm_size = (BLOCK_DIM + 2 * halo) * (BLOCK_DIM + 2 * halo) * channels * sizeof(unsigned char);
    nlmKernel_sm<<<grid, block, sm_size>>>(d_in, d_out, width, height, src.step[0], searchRadius, patchRadius, h * h, channels); // FIXED: Call kernel, not self
    cudaDeviceSynchronize(); 

    cudaMemcpy(dst.data, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out);
}
void fftDeblur(const cv::Mat& src, cv::Mat& dst, double gaussian_sigma_for_deblur, double sharpening_amount) {
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(0, 0), gaussian_sigma_for_deblur); 
    
    cv::addWeighted(src, 1.0 + sharpening_amount, blurred, -sharpening_amount, 0, dst);
}
int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        std::cerr << "Error: No CUDA devices found." << std::endl;
        return 1;
    }
    std::cout << "Found " << devCount << " CUDA devices." << std::endl;

    const std::string input_image_path = "D:\\ddmoo\\images\\data\\20056.jpg"; 
    std::string input_filename = "20056"; 

    cv::Mat src_img = cv::imread(input_image_path, cv::IMREAD_COLOR); 
    if (src_img.empty()) {
        std::cerr << "Error: Could not open or find the image '" << input_image_path << "'." << std::endl;
        std::cerr << "Please ensure the path is correct and the file exists." << std::endl;
        return -1;
    }

    std::cout << "Original image loaded: " << src_img.cols << "x" << src_img.rows << " with " << src_img.channels() << " channels." << std::endl;

    cv::Mat upscaled_img;
    double scale_factor = 2.0; 
    cv::resize(src_img, upscaled_img, cv::Size(), scale_factor, scale_factor, cv::INTER_LANCZOS4);
    cv::imwrite(input_filename + "_upscaled.jpg", upscaled_img);
    std::cout << "Image upscaled to " << upscaled_img.cols << "x" << upscaled_img.rows << ". Saved as '" << input_filename << "_upscaled.jpg'" << std::endl;
    cv::Mat noisy_img = upscaled_img.clone();
    double mean = 0.0; 
    double stddev = 40.0; 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);

    for (int r = 0; r < noisy_img.rows; ++r) {
        for (int c = 0; c < noisy_img.cols; ++c) {
            for (int ch = 0; ch < noisy_img.channels(); ++ch) {
                noisy_img.at<cv::Vec3b>(r, c)[ch] = 
                    cv::saturate_cast<uchar>(upscaled_img.at<cv::Vec3b>(r, c)[ch] + d(gen));
            }
        }
    }
    cv::imwrite(input_filename + "_noisy_upscaled.jpg", noisy_img);
    std::cout << "Gaussian noise added to upscaled image. Saved as '" << input_filename << "_noisy_upscaled.jpg'" << std::endl;
    cv::Mat gaussian_denoised_img;
    cv::Mat gaussian_deblurred_img;
    cv::Mat bilateral_denoised_img;
    cv::Mat nlm_denoised_img;
    cv::Mat nlm_sharpened_img;
    std::cout << "Applying Gaussian Denoising to noisy upscaled image..." << std::endl;
    int ksize_gaussian = 19;
    double sigma_gaussian = 7.0; 
    cudaGaussian_advanced(noisy_img, gaussian_denoised_img, ksize_gaussian, sigma_gaussian);
    cv::imwrite(input_filename + "_gaussian_denoised.jpg", gaussian_denoised_img);
    std::cout << "Gaussian denoising complete. Saved as '" << input_filename << "_gaussian_denoised.jpg'" << std::endl;
    std::cout << "Applying Edge Enhancement (Unsharp Mask) to Gaussian denoised image..." << std::endl;
    fftDeblur(gaussian_denoised_img, gaussian_deblurred_img, 3.0, 1.0); 
    cv::imwrite(input_filename + "_gaussian_deblurred.jpg", gaussian_deblurred_img);
    std::cout << "Gaussian denoised with edge enhancement complete. Saved as '" << input_filename << "_gaussian_deblurred.jpg'" << std::endl;
    std::cout << "Applying Bilateral Denoising to noisy upscaled image..." << std::endl;
    int diameter_bilateral = 49; 
    double sigmaSpace_bilateral = 49.0; 
    double sigmaColor_bilateral = 35.0; 
    cudaBilateral_advanced(noisy_img, bilateral_denoised_img, diameter_bilateral, sigmaSpace_bilateral, sigmaColor_bilateral);
    cv::imwrite(input_filename + "_bilateral_denoised.jpg", bilateral_denoised_img);
    std::cout << "Bilateral denoising complete. Saved as '" << input_filename << "_bilateral_denoised.jpg'" << std::endl;
    std::cout << "Applying Fast Non-Local Means Denoising to noisy upscaled image..." << std::endl;
    int searchRadius_nlm = 30; 
    int patchRadius_nlm = 9;   
    double h_nlm = 80.0; 
    cudaFastNLM_advanced(noisy_img, nlm_denoised_img, searchRadius_nlm, patchRadius_nlm, h_nlm);
    cv::imwrite(input_filename + "_nlm_denoised.jpg", nlm_denoised_img);
    std::cout << "Non-Local Means denoising complete. Saved as '" << input_filename << "_nlm_denoised.jpg'" << std::endl;
    std::cout << "Applying Unsharp Mask (sharpening) to NLM denoised image..." << std::endl; 
    cv::Mat blurred_nlm;
    cv::GaussianBlur(nlm_denoised_img, blurred_nlm, cv::Size(0, 0), 10); 
    cv::addWeighted(nlm_denoised_img, 2.8, blurred_nlm, -1.8, 0, nlm_sharpened_img); 
    cv::imwrite(input_filename + "_nlm_sharpened.jpg", nlm_sharpened_img);
    std::cout << "Sharpening complete. Saved as '" << input_filename << "_nlm_sharpened.jpg'" << std::endl;
    return 0;
}