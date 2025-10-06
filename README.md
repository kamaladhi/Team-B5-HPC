🛰️ Real-Time Satellite and Drone Imagery Analysis Using OpenMP and CUDA
Course: 23AID304 — High Performance and Cloud Computing
Team: Group B5

Members:

Jeevakamal K R – CB.AI.U4AID23115

Jeiesh J S – CB.AI.U4AID23116

Sri Somesh S – CB.AI.U4AID23141

Sai Chakrith – CB.AI.U4AID23143

Suriya Dharsaun K G – CB.AI.U4AID23144

📘 Project Overview

This project focuses on real-time processing of satellite and drone imagery using High Performance Computing (HPC) techniques.
Large-scale aerial datasets demand fast image filtering for feature enhancement, denoising, and segmentation.
Traditional serial image processing fails to meet real-time requirements — thus, parallelization using OpenMP and GPU acceleration using CUDA are applied.

🎯 Objectives

Implement standard image filtering operations (Gaussian, Sobel, Laplacian, Sharpening, Edge Detection).

Accelerate convolution-based filtering using:

OpenMP (CPU parallelism)

CUDA (GPU parallelism)

Measure performance metrics: Execution Time, Speedup, and Efficiency.

Compare Serial vs Parallel (OpenMP) vs GPU (CUDA) implementations.

🧩 System Architecture
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

🧠 Mathematical Background
Convolution Operation

Convolution modifies each pixel based on its neighborhood values:

𝐺
(
𝑥
,
𝑦
)
=
∑
𝑖
=
−
𝑘
𝑘
∑
𝑗
=
−
𝑘
𝑘
𝐹
(
𝑥
−
𝑖
,
𝑦
−
𝑗
)
⋅
𝐾
(
𝑖
,
𝑗
)
G(x,y)=
i=−k
∑
k
	​

j=−k
∑
k
	​

F(x−i,y−j)⋅K(i,j)

Where:

𝐹
(
𝑥
,
𝑦
)
F(x,y) = Input Image

𝐾
(
𝑖
,
𝑗
)
K(i,j) = Kernel/Filter

𝐺
(
𝑥
,
𝑦
)
G(x,y) = Output Image

Each filter kernel (Gaussian, Sobel, etc.) is designed for a specific purpose (e.g., smoothing, edge detection).

🛠️ Tools and Technologies
Category	Tools / Libraries
Programming Languages	C++ (Core), Python (Visualization)
Parallel APIs	OpenMP (CPU), CUDA (GPU)
Libraries	OpenCV, OMP, chrono, filesystem
Dataset	SAT-6 (Kaggle) – 405,000 image patches (28×28×4)
Platform	Windows 10 / Linux (GCC 11+ compatible)
🗂️ Dataset Details

Dataset: SAT-6

Size: 405,000 image patches (28×28 pixels, 4 bands: RGB + NIR)
Classes:

Barren Land

Trees

Grassland

Roads

Buildings

Water Bodies

This dataset provides a real-world benchmark for satellite land-cover classification and filtering performance.

⚙️ Implementation Modules
1️⃣ Serial Implementation

Implements convolution filters in a single-threaded loop.

Uses both cv::Mat and raw pointer arrays for benchmarking.

Baseline for performance comparison.

2️⃣ OpenMP Parallel Implementation

Implements three variants of parallel convolution:

Variant	Description	Key OpenMP Features
Standard	Static division of work (equal rows per thread).	#pragma omp parallel for collapse(2)
Balanced	Dynamic scheduling of work chunks for better load balance.	schedule(dynamic)
Cache-Optimized	Processes image blocks (“tiles”) to fit CPU cache.	Data locality and block tiling
3️⃣ CUDA GPU Implementation (Phase 2)

Uses CUDA kernels with shared memory for parallel filtering.

Provides GPU-accelerated versions of Gaussian and Bilateral filters.

🧮 Filters Implemented
Filter	Purpose
Gaussian Blur	Noise reduction and smoothing.
Sobel (X, Y, Magnitude)	Gradient-based edge detection.
Sharpening Filter	Enhances details and edges.
Laplacian Filter	Detects rapid intensity variation.
Edge Detection	Highlights object boundaries.
🔬 Methodology Summary
Step	Description
1	Load input satellite images (RGB or grayscale).
2	Apply convolution filter (Serial / OpenMP).
3	Measure runtime using chrono and omp timers.
4	Validate output (MSE, PSNR).
5	Record data in .txt and .csv reports.
6	Visualize results and performance plots.
📈 Performance Evaluation
🧾 Variant Performance Summary
Variant	Avg. Time (ms) @8 Threads	Rank
Balanced	7.91 ms	🥇
Cache Optimized	8.00 ms	🥈
Standard	8.56 ms	🥉
Raw Array	8.57 ms	4th
⚡ Thread Scaling (Balanced Variant)
Threads	Avg. Time (ms)	Speedup	Efficiency
1 (Serial)	13.14	1.00×	100%
2	14.85	0.88×	44%
4	7.88	1.67×	41.8%
8	7.91	1.66×	20.8%
16	8.21	1.60×	10%

🧩 Observation:

Optimal performance at 4–8 threads.

Efficiency drops at higher threads due to synchronization overhead.

Balanced scheduling minimizes idle thread time.

🧾 Filter-Wise Performance Highlights
Filter	Serial (ms)	4 Threads	8 Threads	Best Speedup
Gaussian	17.17	26.46	27.62	0.6× (Memory-bound)
Sobel X/Y	~6.0	3.6	2.2	2.7×
Sobel Magnitude	30.44	19.64	17.90	1.7×
Sharpening	19.82	3.63	1.94	10×
Laplacian	6.33	3.65	2.00	3.1×
Edge Detection	6.29	3.51	1.95	3.2×

✅ Inference:

Sharpening and edge filters scale best with parallelization (compute-heavy).

Gaussian blur is limited by memory bandwidth.

🧪 Sample Output Directory Structure
Team-B5-HPC/
│
├── Serial Implementation/
│   ├── Filters/
│   ├── UCMerced_Output_Buildings/
│   │   ├── standard/
│   │   ├── raw_array/
│   │   ├── reports/
│   │   └── serial_performance_data.csv
│
├── OpenMP Implementation/
│   ├── Filters/
│   ├── Headers/
│   ├── convolution_engine_omp.cpp
│   ├── performance_measure_omp.cpp
│   ├── UCMerced_Output_Buildings/
│   │   ├── balanced/
│   │   ├── cache_optimized/
│   │   ├── standard/
│   │   └── reports/
│
└── Results/
    ├── serial_performance_report.txt
    ├── omp_report_threads_4.txt
    ├── omp_report_threads_8.txt
    └── omp_report_threads_16.txt

📉 Visual Performance Summary
⚙️ Speedup vs Threads (Balanced Variant)
Threads	Speedup
1	1.0×
2	0.9×
4	1.7×
8	1.6×
16	1.5×

Speedup saturates beyond 8 threads due to parallel overhead and small input size.

🧮 Efficiency
Threads	Efficiency
2	44%
4	42%
8	21%
16	10%

Efficiency decreases with threads because the image tiles become smaller than cache lines, increasing synchronization cost.

📊 Conclusion

Parallelization with OpenMP successfully improved image processing speed for compute-heavy filters.

Balanced scheduling proved the most effective approach.

Optimal scaling observed up to 4–8 threads.

Memory-bound filters (like Gaussian) benefit more from cache optimization than from extra threads.

Results validate that HPC techniques make real-time satellite image filtering feasible.

🚀 Future Scope

Extend the benchmark to CUDA and MPI implementations for distributed and GPU parallelism.

Apply filters to higher-resolution drone images to better utilize thread scaling.

Integrate real-time visualization and streaming for live satellite feed analysis.

📚 References

OpenMP API Specification 5.0

NVIDIA CUDA Toolkit Documentation

SAT-6 Dataset – Kaggle

OpenCV 4.5 Documentation

🏁 Final Note

This project demonstrates how parallel computing transforms classical image processing into high-speed, scalable pipelines suitable for real-world satellite and UAV applications.

