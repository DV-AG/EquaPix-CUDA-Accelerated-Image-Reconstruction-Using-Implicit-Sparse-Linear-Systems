EquaPix: GPU-Based Image Reconstruction Using Sparse Linear Systems

EquaPix is a CUDA-based grayscale image reconstruction project.

The project treats missing pixels as unknown values in a sparse system based on their neighboring pixels. CUDA is used to update many pixels in parallel.

The project also includes a sequential CPU implementation for correctness comparison and timing.

Project features

- CUDA-based image reconstruction
- Random missing-pixel masks
- Rectangular block masks
- Batch processing of image directories
- Jacobi iteration on the GPU
- CPU reference implementation
- RMSE and PSNR calculation
- CPU/GPU timing comparison
- PNG output for all result images
- Matrix portrait visualization
- Deterministic dataset and masking using fixed seeds

Mathematical model

A grayscale image is treated as a 2D field of pixel values.

For a missing interior pixel:

4*x(i,j) - x(i-1,j) - x(i+1,j) - x(i,j-1) - x(i,j+1) = 0

Known pixels remain fixed.

The complete problem can be represented as:

A*x = b

The matrix is not stored explicitly.

Each missing pixel only depends on its neighboring pixels, so the sparse structure is represented directly using image coordinates.

The image perimeter remains observed so that every missing interior pixel has valid neighboring values.

Jacobi reconstruction

Missing pixels are first initialized using the mean value of the observed pixels.

For every Jacobi iteration, a missing pixel is updated using the average of its four neighbors:

x_new(i,j) =
(x(i-1,j) + x(i+1,j) + x(i,j-1) + x(i,j+1)) / 4

Known pixels stay unchanged.

Two image buffers are used:

- current buffer
- next buffer

After each iteration, the buffers are swapped.

On the GPU, each CUDA thread handles one pixel.

The default experiment uses 500 iterations for random masks. Larger block masks use more iterations because information must propagate farther into the missing region.

Why CUDA

Jacobi iteration performs the same local operation independently across many pixels.

This makes it suitable for GPU parallelism.

CUDA is used for:

- applying the mask
- initializing reconstruction buffers
- Jacobi updates
- squared-error calculation

Images are processed one at a time, while the pixels inside each image are processed in parallel.

Project structure

The main source files are:

src/main.cu
Command-line handling, batch processing, metrics, logging and output.

src/kernels.cu and src/kernels.cuh
CUDA kernels and GPU memory handling.

src/image_io.cpp and src/image_io.h
PGM input handling.

src/png_write.cpp
Writes grayscale result images directly as PNG files.

src/cpu_reference.cpp and src/cpu_reference.h
CPU Jacobi implementation, mask generation and matrix portrait creation.

src/metrics.cpp and src/metrics.h
RMSE, PSNR and CPU/GPU comparison functions.

scripts/generate_dataset.py
Generates the synthetic image dataset.

scripts/run_experiments.sh
Runs the complete experiment set.

tests/
Contains host, dataset, PNG and CUDA tests.

More numerical details are available in docs/architecture.md.

Build

The project requires:

- NVIDIA CUDA toolkit
- nvcc
- C++17-compatible compiler
- make
- Python 3

Build the project using:

make

Check the executable:

./equapix --help

Run host tests:

make test

Run CUDA tests:

make gpu-test

To clean the build:

make clean

Dataset

Generate the standard dataset using:

python3 scripts/generate_dataset.py --count 200 --width 256 --height 256 --seed 42

This creates images inside:

data/generated/

The generated dataset contains several pattern types, including:

- gradients
- circles
- rectangles
- checkerboards
- sine-wave patterns
- diagonal patterns
- combined geometry
- smooth regions
- textured patterns

The same seed and dimensions generate the same dataset.

A small sample dataset is also included in:

data/sample/

Running one experiment

Example with a 20% random mask:

./equapix --input data/generated --output results/random_20 
  --mask random --missing-percent 20 --iterations 500 
  --seed 42 --cpu-benchmark-count 10

Example with a rectangular missing block:

./equapix --input data/sample --output results/block_64 
  --mask block --block-size 64 --iterations 1500 
  --cpu-benchmark-count 8

--cpu-benchmark-count controls how many images are also reconstructed using the sequential CPU implementation.

Setting:

--cpu-benchmark-count 0

disables CPU benchmarking.

Running the full experiment

From the project directory:

bash scripts/run_experiments.sh

The script:

- builds the project
- runs tests
- generates the 200-image dataset
- runs 5% random masking
- runs 10% random masking
- runs 20% random masking
- runs 30% random masking
- runs the block-mask experiment
- checks the generated outputs

The main experiment folders are:

results/random_5/
results/random_10/
results/random_20/
results/random_30/
results/block_64/

Output files

Each experiment folder contains:


execution_log.txt
examples/
matrix_portraits/

For each selected image, the examples directory contains:

<image>_original.png
<image>_masked.png
<image>_reconstructed.png

For example:

image_0000_gradient_original.png
image_0000_gradient_masked.png
image_0000_gradient_reconstructed.png

These PNG files can be opened directly in the Coursera Lab file browser.

No Pillow, ImageMagick, OpenCV or separate conversion step is needed.

Input images remain in PGM format, while all user-facing results are written as PNG.

Metrics

The program records several values in execution_log.txt of each folder of results(created after the run).

Important columns include:

requested_missing_percent
Requested random missing percentage.

actual_missing_percent
Actual percentage of missing pixels.

iterations
Number of Jacobi iterations.

gpu_ms
GPU Jacobi execution time.

gpu_pipeline_ms
GPU processing time including setup and transfers.

total_ms
Overall processing time.

rmse
Whole-image RMSE.

psnr
Whole-image PSNR.

missing_rmse
RMSE over missing pixels only.

missing_psnr
PSNR over missing pixels only.

cpu_ms
CPU Jacobi execution time.

speedup
CPU time divided by GPU time.

cpu_gpu_max_abs
Maximum CPU/GPU pixel difference.

validation
CPU/GPU comparison result.

RMSE is calculated as:

RMSE = sqrt(sum((original - reconstructed)^2) / pixel_count)

PSNR is calculated as:

PSNR = 20 * log10(255 / RMSE)

The missing-region metrics are useful because observed pixels remain unchanged and therefore reduce whole-image error.

CPU and GPU comparison

The CPU and GPU implementations use:

- the same mask
- the same initialization
- the same iteration count

The outputs are compared using the maximum absolute pixel difference.

The CPU implementation is used mainly for validation and timing comparison.

The reconstruction itself is performed using CUDA.

Matrix portrait

Each experiment can also generate a small PNG visualization of the sparse system.

The portrait represents a small sampled pixel grid.

Diagonal marks represent equations, while neighboring marks show connections between adjacent pixels.

The full image matrix is never explicitly allocated.

The reconstruction uses the local pixel-neighbor relationships directly.

Limitations

Jacobi reconstruction performs harmonic interpolation.

It works well when missing values can be estimated from nearby pixels, especially in smooth regions.

It does not reconstruct complex missing texture or detailed objects in the same way as machine-learning image inpainting.

Large missing regions may also require more Jacobi iterations.

The current version:

- works on grayscale images
- uses a fixed iteration count
- processes one image at a time
- does not use a convergence stopping condition
- does not use edge-aware or learned reconstruction

Possible extensions include convergence-based stopping, RGB support, multigrid methods, red-black Gauss-Seidel and overlapping data transfers using CUDA streams.

Repository structure

EquaPix/
README.md
Makefile
LICENSE
src/
scripts/
tests/
data/
data/sample/
docs/
docs/architecture.md
results/
