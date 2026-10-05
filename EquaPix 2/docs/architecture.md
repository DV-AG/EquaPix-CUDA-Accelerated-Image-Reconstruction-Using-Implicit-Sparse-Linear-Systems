# Architecture and Numerical Model

Data flow

The program follows this basic pipeline:

PGM input image → create mask → CUDA reconstruction → calculate errors → save PNG results and CSV metrics.

The program processes one image at a time. Each image is reconstructed in parallel on the GPU, where CUDA threads work on different pixels.

The project does not create a large dense matrix. The sparse system is represented using the relationships between neighboring pixels.

Sparse system

Each pixel is treated as one variable.

For a known pixel, its value stays fixed.

For a missing interior pixel, the reconstruction uses its four neighbors:

4x(i,j) - x(i-1,j) - x(i+1,j) - x(i,j-1) - x(i,j+1) = 0

This can be viewed as a sparse linear system:

A x = b

The matrix is never explicitly stored because each pixel only depends on a small number of neighboring pixels.

The outer boundary of the image is kept known so that every missing interior pixel can safely use its four neighbors.

**Masking**

The project supports two types of masks.

Random masking removes a chosen percentage of interior pixels.

Block masking removes a rectangular region near the center of the image.

A fixed seed is used so that the same experiments can be reproduced.

**Jacobi reconstruction**

Missing pixels are first initialized using the mean value of the observed pixels.

Jacobi iteration is then performed for a fixed number of iterations.

During each iteration, every missing pixel is updated using the average of its four neighbors from the previous iteration.

Known pixels remain unchanged.

Two image buffers are used:

current buffer  
next buffer

After every iteration, the buffers are swapped.

On the GPU, different CUDA threads update different pixels in parallel.

The algorithm performs harmonic interpolation. It works well for smooth missing regions, but it cannot perfectly recreate texture or detailed objects that are completely missing.

**CPU and GPU versions**

The project contains both a sequential CPU implementation and a CUDA implementation.

The CPU implementation is mainly used to check correctness and measure speedup.

Both versions:

- use the same mask
- use the same initialization
- perform the same number of Jacobi iterations

The GPU and CPU outputs are compared to make sure their results are sufficiently close.

**Metrics**

The project calculates:

- RMSE
- PSNR
- missing-region RMSE
- missing-region PSNR
- CPU execution time
- GPU execution time
- CPU/GPU speedup

RMSE and PSNR compare the reconstructed image with the original image.

The missing-region metrics are especially useful because known pixels are unchanged and would otherwise make the whole-image error appear smaller.

GPU timing is measured using CUDA events.

Matrix portrait

The project also creates a small matrix portrait showing the sparse structure of the reconstruction problem.

It uses a small representative image grid rather than constructing the full matrix.

The portrait shows that each unknown pixel interacts mainly with itself and its neighboring pixels.

The actual reconstruction still uses the implicit pixel-neighbor representation and never allocates a full N × N matrix.

Output files

For each processed image, the program saves:

- original image
- masked image
- reconstructed image
- matrix portrait

All user-facing images are saved directly as PNG files.

The experiment folder also contains:

- metrics.csv
- execution_log.txt

Typical result folders are:

results/random_5/  
results/random_10/  
results/random_20/  
results/random_30/  
results/block_64/

PNG output

Input images remain in PGM format because they are simple to load without external libraries.

All result images are written directly as PNG using the project's C++ PNG writer.
