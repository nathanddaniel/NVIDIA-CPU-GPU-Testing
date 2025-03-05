# NVIDIA-CPU-GPU-Testing

This repository contains CUDA programs for learning and experimenting with GPU programming using NVIDIA's CUDA framework. The repository includes:

1. **Tutorial Code from Appendix**  
   - A modified version of `asyncAPI.cu` that demonstrates **asynchronous execution** in CUDA.
   - The example shows **how the CPU and GPU can work in parallel** using asynchronous memory transfers and event timing.

2. **CUDA Device Query**  
   - A CUDA program to **query and display GPU device properties**.
   - This provides details such as **GPU name, memory size, compute capability, number of cores, and multiprocessors**.

3. **Matrix Multiplication Using CUDA**  
   - A CUDA implementation of **matrix multiplication (C = A × B)** using **parallel thread execution**.
   - Demonstrates **block and thread indexing**, **shared memory usage**, and **performance comparison** with CPU computation.