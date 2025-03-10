#include <stdio.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <time.h>

// CPU Matrix Multiplication
__host__ void matrixMulCPU(float* P, float* M, float* N, int Nsize) {
    for (int i = 0; i < Nsize; i++) {
        for (int j = 0; j < Nsize; j++) {
            float sum = 0;
            for (int k = 0; k < Nsize; k++) {
                sum += M[i * Nsize + k] * N[k * Nsize + j];
            }
            P[i * Nsize + j] = sum;
        }
    }
}

// GPU Matrix Multiplication Kernel (Parallel Execution)
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

// Function to measure Host → Device transfer time
float CPUtoGPUTime(float* d_ptr, float* h_ptr, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

// Function to measure Device → Host transfer time
float GPUtoCPUTime(float* h_ptr, float* d_ptr, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int floatSize = sizeof(float);

    for (int i = 0; i < numOfSizes; i++) {
        int N = matrixSizes[i];
        
        // Only execute for the required matrix sizes
        if (N == 256 || N == 512 || N == 1024) {
            int bytes = N * N * floatSize;

            // Allocate memory on CPU
            float *h_M = (float*)malloc(bytes);
            float *h_N = (float*)malloc(bytes);
            float *h_P_cpu = (float*)malloc(bytes);
            float *h_P_gpu = (float*)malloc(bytes);

            // Initialize matrices with random values
            for (int j = 0; j < N * N; j++) {
                h_M[j] = (float)(rand() % 5);
                h_N[j] = (float)(rand() % 5);
            }

            // Allocate memory on GPU
            float *d_M, *d_N, *d_P;
            cudaMalloc(&d_M, bytes);
            cudaMalloc(&d_N, bytes);
            cudaMalloc(&d_P, bytes);

            // Measure CPU execution time
            clock_t start_cpu = clock();
            matrixMulCPU(h_P_cpu, h_M, h_N, N);
            clock_t end_cpu = clock();
            float cpu_exec_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;

            // Measure Host → Device transfer time
            float h2d_time_M = CPUtoGPUTime(d_M, h_M, bytes);
            float h2d_time_N = CPUtoGPUTime(d_N, h_N, bytes);
            float total_h2d_time = h2d_time_M + h2d_time_N;

            dim3 threadsPerBlock(1,1);
            dim3 numBlocks(1,1);

            // Measure GPU execution time (same approach as CPU)
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            MatrixMulKernel<<<numBlocks, threadsPerBlocks>>>(d_M, d_N, d_P, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float gpu_exec_time = 0;
            cudaEventElapsedTime(&gpu_exec_time, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
    
            // Measure Device → Host transfer time
            float d2h_time_P = GPUtoCPUTime(h_P_gpu, d_P, bytes);

            // Print results
            printf("\nMatrix Size: %d x %d\n", N, N);
            printf("Host to Device Transfer Time: %.3f ms\n", total_h2d_time);
            printf("GPU Execution Time: %.3f ms\n", gpu_exec_time);
            printf("Device to Host Transfer Time: %.3f ms\n", d2h_time_P);
            printf("CPU Execution Time: %.3f ms\n", cpu_exec_time);

            // Free memory
            free(h_M);
            free(h_N);
            free(h_P_cpu);
            free(h_P_gpu);
            cudaFree(d_M);
            cudaFree(d_N);
            cudaFree(d_P);
        }
    }

    return 0;
}