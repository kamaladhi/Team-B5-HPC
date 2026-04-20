# 🛰️ Comparative Performance Analysis of Multi-Core and GPU-Accelerated Convolution Engines for Large-Scale Remote Sensing Imagery

**Course:** 23AID304 — High Performance and Cloud Computing  
**Team:** Group B5

## 👥 Members
- Jeevakamal K R – CB.AI.U4AID23115
- Jeiesh J S – CB.AI.U4AID23116
- Sri Somesh S – CB.AI.U4AID23141
- Sai Chakrith – CB.AI.U4AID23143
- Suriya Dharsaun K G – CB.AI.U4AID23144

---

## 📘 Project Overview

This project focuses on real-time processing of satellite and drone imagery using High Performance Computing (HPC) techniques.  
Large-scale aerial datasets demand fast image filtering for feature enhancement, denoising, and segmentation.  
Traditional serial image processing fails to meet real-time requirements — thus, parallelization using OpenMP and GPU acceleration using CUDA are applied.

---

## 🎯 Objectives

- Implement standard image filtering operations (Gaussian, Sobel, Laplacian, Sharpening, Edge Detection).
- Accelerate convolution-based filtering using:
  - OpenMP (CPU parallelism)
  - CUDA (GPU parallelism)
- Measure performance metrics: Execution Time, Speedup, and Efficiency.
- Compare Serial vs Parallel (OpenMP) vs GPU (CUDA) implementations.

---

## 🧩 System Architecture

```
Input Image Dataset (SAT-6)
        │
        ▼
 ┌──────────────────────────┐
 │   Convolution Engine     │
 │ (Serial / OpenMP / CUDA) │
 └──────────┬───────────────┘
            │
  ┌─────────┴─────────┐
  │                   │
Serial           OpenMP Parallel
(Baseline)       (Static / Balanced / Cache Optimized)
  │                   │
  └─────────┬─────────┘
            ▼
 ┌─────────────────────────────┐
 │ Performance Measurement OMP │
 │ (Timing, MSE, PSNR, Speedup)│
 └─────────────────────────────┘
            │
            ▼
  Visualized Output & Report
```

---

## 🧠 Mathematical Background

**Convolution Operation:**  
Each pixel is modified based on its neighborhood values:

```
G(x, y) = Σ_{i=-k}^{k} Σ_{j=-k}^{k} F(x-i, y-j) * K(i, j)
```
- F(x, y): Input Image
- K(i, j): Kernel/Filter
- G(x, y): Output Image

Each filter kernel (Gaussian, Sobel, etc.) is designed for a specific purpose (e.g., smoothing, edge detection).

---

## 🛠️ Tools and Technologies

| Category              | Tools / Libraries                        |
|-----------------------|------------------------------------------|
| Programming Languages | C++ (Core), Python (Visualization)       |
| Parallel APIs         | OpenMP (CPU), CUDA (GPU)                 |
| Libraries             | OpenCV, OMP, chrono, filesystem          |
| Dataset               | SAT-6 (Kaggle) – 405,000 image patches   |
| Platform              | Windows 10 / Linux (GCC 11+ compatible)  |

---

## 🗂️ Dataset Details

- **Dataset:** SAT-6
- **Size:** 405,000 image patches (28×28 pixels, 4 bands: RGB + NIR)
- **Classes:** Barren Land, Trees, Grassland, Roads, Buildings, Water Bodies

---

## ⚙️ Implementation Modules

### 1️⃣ Serial Implementation
- Single-threaded convolution filters.
- Uses both `cv::Mat` and raw pointer arrays.
- Baseline for performance comparison.

### 2️⃣ OpenMP Parallel Implementation
Three variants:
| Variant         | Description                                      | Key OpenMP Features                |
|-----------------|--------------------------------------------------|------------------------------------|
| Standard        | Static division of work (equal rows per thread)  | `#pragma omp parallel for collapse(2)` |
| Balanced        | Dynamic scheduling for better load balance       | `schedule(dynamic)`                |
| Cache-Optimized | Processes image blocks (“tiles”) for cache usage | Data locality and block tiling     |

### 3️⃣ CUDA GPU Implementation
- GPU acceleration via CUDA Kernels for high-throughput image convolution.
- Memory-optimized data transfer between Host (CPU) and Device (GPU).
- Integrated Support for Gaussian, Sobel, and Sharpening filters at scale.

---

## 🧮 Filters Implemented

| Filter            | Purpose                        |
|-------------------|-------------------------------|
| Gaussian Blur     | Noise reduction and smoothing  |
| Sobel (X, Y, Mag) | Gradient-based edge detection  |
| Sharpening        | Enhances details and edges     |
| Laplacian         | Detects rapid intensity change |
| Edge Detection    | Highlights object boundaries   |

---

## 🔬 Methodology Summary

1. Load input satellite images (RGB or grayscale).
2. Apply convolution filter (Serial / OpenMP).
3. Measure runtime using chrono and omp timers.
4. Validate output (MSE, PSNR).
5. Record data in .txt and .csv reports.
6. Visualize results and performance plots.

---

## 📈 Performance Evaluation

### 🧾 Variant Performance Summary (@8 Threads)
| Variant         | Avg. Time (ms) | Rank |
|-----------------|---------------|------|
| Balanced        | 7.91          | 🥇   |
| Cache Optimized | 8.00          | 🥈   |
| Standard        | 8.56          | 🥉   |
| Raw Array       | 8.57          | 4th  |

### ⚡ Thread Scaling (Balanced Variant)
| Threads | Avg. Time (ms) | Speedup | Efficiency |
|---------|---------------|---------|------------|
| 1       | 13.14         | 1.00×   | 100%       |
| 2       | 14.85         | 0.88×   | 44%        |
| 4       | 7.88          | 1.67×   | 41.8%      |
| 8       | 7.91          | 1.66×   | 20.8%      |
| 16      | 8.21          | 1.60×   | 10%        |

**Observation:**  
- Optimal performance at 4–8 threads.
- Efficiency drops at higher threads due to synchronization overhead.
- Balanced scheduling minimizes idle thread time.

### 🧾 Filter-Wise Performance Highlights

| Filter         | Serial (ms) | 4 Threads | 8 Threads | Best Speedup         |
|----------------|-------------|-----------|-----------|----------------------|
| Gaussian       | 17.17       | 26.46     | 27.62     | 0.6× (Memory-bound)  |
| Sobel X/Y      | ~6.0        | 3.6       | 2.2       | 2.7×                |
| Sobel Magnitude| 30.44       | 19.64     | 17.90     | 1.7×                |
| Sharpening     | 19.82       | 3.63      | 1.94      | 10×                 |
| Laplacian      | 6.33        | 3.65      | 2.00      | 3.1×                |
| Edge Detection | 6.29        | 3.51      | 1.95      | 3.2×                |

**Inference:**  
- Sharpening and edge filters scale best (compute-heavy).
- Gaussian blur is limited by memory bandwidth.

---

## 🗄️ Repository Structure

```
HPC-Satellite-Convolution/
├── CUDA/                       # GPU-accelerated kernels and CUDA source
├── open_mp_implementation/      # Multi-threaded CPU implementation
│   ├── Filters/                 # OpenMP-specific filter logic
│   ├── Framework/               # Benchmarking and driver code
│   └── Report_of_the_open_mp/   # Performance data and scaling reports
├── Serial_Implementation/       # Single-threaded baseline implementation
│   ├── Filters/                 # Standard C++ filter logic
│   ├── Framework/               # Serial benchmark drivers
│   └── Header/                  # Shared serial header files
└── README.md                    # Project documentation
```

---

## 📉 Visual Performance Summary

### ⚙️ Speedup vs Threads (Balanced Variant)
| Threads | Speedup |
|---------|---------|
| 1       | 1.0×    |
| 2       | 0.9×    |
| 4       | 1.7×    |
| 8       | 1.6×    |
| 16      | 1.5×    |

Speedup saturates beyond 8 threads due to parallel overhead and small input size.

### 🧮 Efficiency
| Threads | Efficiency |
|---------|------------|
| 2       | 44%        |
| 4       | 42%        |
| 8       | 21%        |
| 16      | 10%        |

Efficiency decreases with threads because the image tiles become smaller than cache lines, increasing synchronization cost.

---

## 📊 Conclusion

- Parallelization with OpenMP improved image processing speed for compute-heavy filters.
- Balanced scheduling proved most effective.
- Optimal scaling observed up to 4–8 threads.
- Memory-bound filters (like Gaussian) benefit more from cache optimization than extra threads.
- HPC techniques make real-time satellite image filtering feasible.

---

## 🚀 Future Scope

- Extend benchmark to MPI (Message Passing Interface) for distributed clusters.
- Apply filters to higher-resolution drone images and 4K satellite feeds.
- Integrate real-time visualization dashboards for live UAV streaming missions.

---

## 📚 References

- OpenMP API Specification 5.0
- NVIDIA CUDA Toolkit Documentation
- SAT-6 Dataset – Kaggle
- OpenCV 4.5 Documentation

---

## 🏁 Final Note

This project demonstrates how parallel computing transforms classical image processing into high-speed, scalable pipelines suitable for real-world satellite and UAV applications.

