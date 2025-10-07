import torch
import torch.nn as nn
import numpy as np
from torch.utils.cpp_extension import load_inline
import time
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 3x3 convolution kernel
__global__ void conv2d_3x3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_ch, int out_ch, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_ch * h * w;
    
    if (idx < total) {
        int x = idx % w;
        int y = (idx / w) % h;
        int oc = (idx / (w * h)) % out_ch;
        int b = idx / (w * h * out_ch);
        
        float sum = bias[oc];
        
        #pragma unroll
        for (int ic = 0; ic < in_ch; ic++) {
            #pragma unroll
            for (int ky = 0; ky < 3; ky++) {
                #pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    int iy = y + ky - 1;
                    int ix = x + kx - 1;
                    
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                        int in_idx = ((b * in_ch + ic) * h + iy) * w + ix;
                        int w_idx = ((oc * in_ch + ic) * 3 + ky) * 3 + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

// Fused BatchNorm + ReLU
__global__ void batchnorm_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    int batch, int channels, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    
    if (idx < total) {
        int c = (idx / (h * w)) % channels;
        float normalized = (input[idx] - mean[c]) * rsqrtf(var[c] + 1e-5f);
        float result = gamma[c] * normalized + beta[c];
        output[idx] = fmaxf(0.0f, result);  // ReLU
    }
}

// Gaussian blur kernel (for creating training data)
__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int h, int w,
    float sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    
    if (idx < total) {
        int x = idx % w;
        int y = (idx / w) % h;
        int c = (idx / (w * h)) % channels;
        int b = idx / (w * h * channels);
        
        // 5x5 Gaussian kernel
        float kernel[5][5];
        float sum_kernel = 0.0f;
        int ksize = 5;
        int center = ksize / 2;
        
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                float dx = kx - center;
                float dy = ky - center;
                kernel[ky][kx] = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
                sum_kernel += kernel[ky][kx];
            }
        }
        
        // Normalize kernel
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                kernel[ky][kx] /= sum_kernel;
            }
        }
        
        // Apply blur
        float result = 0.0f;
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                int iy = y + ky - center;
                int ix = x + kx - center;
                
                if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                    int in_idx = ((b * channels + c) * h + iy) * w + ix;
                    result += input[in_idx] * kernel[ky][kx];
                }
            }
        }
        output[idx] = result;
    }
}

// Motion blur kernel
__global__ void motion_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int h, int w,
    int blur_length, float angle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    
    if (idx < total) {
        int x = idx % w;
        int y = (idx / w) % h;
        int c = (idx / (w * h)) % channels;
        int b = idx / (w * h * channels);
        
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int i = -blur_length/2; i <= blur_length/2; i++) {
            int dy = (int)(i * sin_angle);
            int dx = (int)(i * cos_angle);
            int iy = y + dy;
            int ix = x + dx;
            
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                int in_idx = ((b * channels + c) * h + iy) * w + ix;
                sum += input[in_idx];
                count++;
            }
        }
        
        output[idx] = (count > 0) ? (sum / count) : input[idx];
    }
}

// Forward declarations
torch::Tensor conv2d_3x3_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);

torch::Tensor batchnorm_relu_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var
);

torch::Tensor gaussian_blur_forward(
    torch::Tensor input,
    float sigma
);

torch::Tensor motion_blur_forward(
    torch::Tensor input,
    int blur_length,
    float angle
);

// Implementation
torch::Tensor conv2d_3x3_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch = input.size(0);
    auto in_ch = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);
    auto out_ch = weight.size(0);
    
    auto output = torch::zeros({batch, out_ch, h, w}, input.options());
    
    int total = batch * out_ch * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv2d_3x3_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_ch, out_ch, h, w
    );
    
    return output;
}

torch::Tensor batchnorm_relu_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var
) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total = batch * channels * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    batchnorm_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch, channels, h, w
    );
    
    return output;
}

torch::Tensor gaussian_blur_forward(
    torch::Tensor input,
    float sigma
) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total = batch * channels * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    gaussian_blur_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, h, w, sigma
    );
    
    return output;
}

torch::Tensor motion_blur_forward(
    torch::Tensor input,
    int blur_length,
    float angle
) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total = batch * channels * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    motion_blur_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, h, w, blur_length, angle
    );
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor conv2d_3x3_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);

torch::Tensor batchnorm_relu_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var
);

torch::Tensor gaussian_blur_forward(
    torch::Tensor input,
    float sigma
);

torch::Tensor motion_blur_forward(
    torch::Tensor input,
    int blur_length,
    float angle
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_3x3_forward", &conv2d_3x3_forward, "Custom 3x3 Conv2D forward");
    m.def("batchnorm_relu_forward", &batchnorm_relu_forward, "Fused BatchNorm+ReLU forward");
    m.def("gaussian_blur_forward", &gaussian_blur_forward, "Gaussian blur");
    m.def("motion_blur_forward", &motion_blur_forward, "Motion blur");
}
"""
print("Compiling C++/CUDA extensions for deblurring...")
print("This may take 1-2 minutes on first run...")

try:
    cuda_module = load_inline(
        name='unet_deblur_kernels',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['conv2d_3x3_forward', 'batchnorm_relu_forward', 
                   'gaussian_blur_forward', 'motion_blur_forward'],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math']
    )
    print("✓ CUDA extensions compiled successfully!")
except Exception as e:
    print(f"✗ Failed to compile CUDA extensions: {e}")
    print("Falling back to PyTorch operations...")
    cuda_module = None

def apply_gaussian_blur(images, sigma=2.0, use_cuda=True):
    """Apply Gaussian blur to images"""
    if use_cuda and cuda_module is not None:
        return cuda_module.gaussian_blur_forward(images, sigma)
    else:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur = nn.functional.avg_pool2d(
            images, kernel_size=kernel_size, stride=1, 
            padding=kernel_size//2
        )
        return blur

def apply_motion_blur(images, blur_length=15, angle=None, use_cuda=True):
    """Apply motion blur to images"""
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    
    if use_cuda and cuda_module is not None:
        return cuda_module.motion_blur_forward(images, blur_length, angle)
    else:
        return images 

def apply_mixed_blur(images, blur_type='gaussian'):
    """Apply various types of blur"""
    if blur_type == 'gaussian':
        sigma = np.random.uniform(1.5, 3.5)
        return apply_gaussian_blur(images, sigma)
    elif blur_type == 'motion':
        length = np.random.randint(10, 20)
        angle = np.random.uniform(0, 2 * np.pi)
        return apply_motion_blur(images, length, angle)
    else:
        if np.random.rand() < 0.5:
            return apply_gaussian_blur(images, np.random.uniform(1.5, 3.0))
        else:
            return apply_motion_blur(images, np.random.randint(10, 18))
class CustomDoubleConv(nn.Module):
    """Double convolution block using custom CUDA kernels"""
    
    def __init__(self, in_channels, out_channels, use_cuda_kernel=True):
        super(CustomDoubleConv, self).__init__()
        self.use_cuda_kernel = use_cuda_kernel and cuda_module is not None
        self.conv1_weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        self.conv1_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn1_gamma = nn.Parameter(torch.ones(out_channels))
        self.bn1_beta = nn.Parameter(torch.zeros(out_channels))
        self.bn1_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.bn1_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.conv2_weight = nn.Parameter(torch.randn(out_channels, out_channels, 3, 3))
        self.conv2_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn2_gamma = nn.Parameter(torch.ones(out_channels))
        self.bn2_beta = nn.Parameter(torch.zeros(out_channels))
        self.bn2_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.bn2_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_weight, mode='fan_out', nonlinearity='relu')
        if not self.use_cuda_kernel:
            self.fallback = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        if self.use_cuda_kernel:
            x = cuda_module.conv2d_3x3_forward(x, self.conv1_weight, self.conv1_bias)
            x = cuda_module.batchnorm_relu_forward(
                x, self.bn1_gamma, self.bn1_beta, self.bn1_mean, self.bn1_var
            )
            x = cuda_module.conv2d_3x3_forward(x, self.conv2_weight, self.conv2_bias)
            x = cuda_module.batchnorm_relu_forward(
                x, self.bn2_gamma, self.bn2_beta, self.bn2_mean, self.bn2_var
            )
            return x
        else:
            return self.fallback(x)
class UNetDeblur(nn.Module):
    """U-Net for image deblurring and sharpening"""
    
    def __init__(self, in_channels=3, out_channels=3, use_cuda_kernels=True):
        super(UNetDeblur, self).__init__()
        self.enc_conv1 = CustomDoubleConv(in_channels, 32, use_cuda_kernels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = CustomDoubleConv(32, 64, use_cuda_kernels)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = CustomDoubleConv(64, 128, use_cuda_kernels)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = CustomDoubleConv(128, 256, use_cuda_kernels)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = CustomDoubleConv(256, 128, use_cuda_kernels)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = CustomDoubleConv(128, 64, use_cuda_kernels)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = CustomDoubleConv(64, 32, use_cuda_kernels)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        skip1 = self.enc_conv1(x)
        x = self.pool1(skip1)
        skip2 = self.enc_conv2(x)
        x = self.pool2(skip2)
        skip3 = self.enc_conv3(x)
        x = self.pool3(skip3)
        x = self.bottleneck(x)
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec_conv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec_conv2(x)
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec_conv1(x)
        x = self.out_conv(x)
        return self.sigmoid(x)
from torch.utils.data import Dataset, DataLoader

class BuildingsDataset(Dataset):
    def __init__(self, root_dir, start_idx, end_idx, img_size=256):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.image_paths = []
        
        for i in range(start_idx, end_idx + 1):
            img_path = self.root_dir / f"buildings{i:02d}.tif"
            if img_path.exists():
                self.image_paths.append(img_path)
        
        print(f"  Loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, 0
def train_model():
    """Training loop for deblurring model"""
    
    DATA_PATH = r"D:\UCMerced_LandUse\UCMerced_LandUse\Images\buildings"
    BATCH_SIZE = 4
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    BLUR_SIGMA = 2.5  
    IMG_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("U-NET DEBLURRING WITH C++/CUDA ACCELERATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"CUDA Kernels: {'Enabled' if cuda_module else 'Disabled (using PyTorch)'}")
    print(f"Task: Blur → Sharp (Deblurring + Sharpening)")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Blur Sigma: {BLUR_SIGMA}")
    print("="*70)
    train_dataset = BuildingsDataset(DATA_PATH, 0, 79, img_size=IMG_SIZE)
    test_dataset = BuildingsDataset(DATA_PATH, 80, 99, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = UNetDeblur(use_cuda_kernels=(cuda_module is not None)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting deblurring training...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sharp_images, _) in enumerate(train_loader):
            sharp_images = sharp_images.to(device)
            blurred_images = apply_gaussian_blur(sharp_images, sigma=BLUR_SIGMA)
            optimizer.zero_grad()
            outputs = model(blurred_images)
            loss = criterion(outputs, sharp_images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sharp_images.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}]", 
                      end='\r', flush=True)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sharp_images, _ in test_loader:
                sharp_images = sharp_images.to(device)
                blurred_images = apply_gaussian_blur(sharp_images, sigma=BLUR_SIGMA)
                outputs = model(blurred_images)
                loss = criterion(outputs, sharp_images)
                val_loss += loss.item() * sharp_images.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_deblur_model.pth')
            print(f"  >>> Best model saved!")
    
    print("="*70)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses
def visualize_deblurring(model, data_path, device, num_samples=4):
    """Visualize deblurring results"""
    
    dataset = BuildingsDataset(data_path, 0, 99, img_size=256)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    
    model.eval()
    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            sharp_img, _ = dataset[img_idx]
            sharp_img = sharp_img.unsqueeze(0).to(device)
            blurred_img = apply_gaussian_blur(sharp_img, sigma=2.5)
            deblurred_img = model(blurred_img)
            blurred_np = blurred_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            deblurred_np = deblurred_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            sharp_np = sharp_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            blurred_np = np.clip(blurred_np, 0, 1)
            deblurred_np = np.clip(deblurred_np, 0, 1)
            sharp_np = np.clip(sharp_np, 0, 1)
            
            axes[idx, 0].imshow(blurred_np)
            axes[idx, 0].set_title(f"Blurred (Image {img_idx})", fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(deblurred_np)
            axes[idx, 1].set_title(f"Sharpened (Restored)", fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(sharp_np)
            axes[idx, 2].set_title(f"Original (Sharp)", fontweight='bold')
            axes[idx, 2].axis('off')
    
    plt.suptitle('Deblurring Results: Blur → Sharp', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('deblurring_results.png', dpi=150, bbox_inches='tight')
    print("\nDeblurring visualization saved: deblurring_results.png")
    plt.show()


#MAIN

if __name__ == "__main__":
    print("\n" + "="*70)
    print("U-NET DEBLURRING WITH PYTHON + C++/CUDA")
    print("Task: Convert Blurred Images to Sharp, Clear Images")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available")
    
    # Training our model
    model, train_losses, val_losses = train_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_deblurring(model, r"D:\UCMerced_LandUse\UCMerced_LandUse\Images\buildings", 
                        device, num_samples=4)
    
    # Plot tthe graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', marker='o', markersize=3)
    plt.plot(val_losses, label='Validation', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Deblurring Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train', marker='o', markersize=3)
    plt.plot(val_losses, label='Validation', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training History (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_deblur.png', dpi=150)
    print("Training plot saved: training_deblur.png")
    plt.show()
    
    print("\n" + "="*70)
    print("DEBLURRING COMPLETE!")
    print("="*70)
    print("Outputs:")
    print("  1. Model: best_deblur_model.pth")
    print("  2. Results: deblurring_results.png")
    print("  3. Training: training_deblur.png")
    print("="*70)
    print("\n✓ Done!")
