#include <stdio.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <cmath>

// Matrix Sizes
#define BLOCK_WIDTH 32

// CPU Matrix Multiplication
void matrixMulCPU(float* P, float* M, float* N, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0;
            for (int i = 0; i < width; i++) {
                sum += M[row * width + i] * N[i * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

// GPU Kernel for Matrix Multiplication
__global__ void MatrixMulKernel(float* M, float* N, float* P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0;
        for (int i = 0; i < width; i++) {
            sum += M[row * width + i] * N[i * width + col];
        }
        P[row * width + col] = sum;
    }
}

int main() {
    // Array of matrix sizes to test
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int floatSize = sizeof(float);

    for (int i = 0; i < numOfSizes; i++) {
        int N = matrixSizes[i];
        int bytes = N * N * floatSize;

        // Allocate memory on CPU
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P_cpu = (float*)malloc(bytes);
        float* h_P_gpu = (float*)malloc(bytes);
        
        // Initialize matrices with random values
        for (int j = 0; j < N * N; j++) {
            h_M[j] = (float)(rand() % 5);
            h_N[j] = (float)(rand() % 5);
        }

        // Allocate memory on GPU
        float* d_M, *d_N, *d_P;
        cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_N, bytes);
        cudaMalloc(&d_P, bytes);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // **Measure Host to Device Transfer Time**
        float h2d_time;
        cudaEventRecord(start);
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&h2d_time, start, stop);
        printf("\nMatrix Size: %d x %d\n", N, N);
        printf("Host to Device Transfer Time: %.3f ms\n", h2d_time);

        // Define grid and block dimensions
        dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 numBlocks((N + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 
                       (N + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

        // **Measure GPU Execution Time**
        float gpu_exec_time;
        cudaEventRecord(start);
        MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_P, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_exec_time, start, stop);
        printf("GPU Execution Time: %.3f ms\n", gpu_exec_time);

        // **Measure Device to Host Transfer Time**
        float d2h_time;
        cudaEventRecord(start);
        cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&d2h_time, start, stop);
        printf("Device to Host Transfer Time: %.3f ms\n", d2h_time);

        // **Total GPU Processing Time (including data transfers)**
        printf("Total GPU Time (including transfers): %.3f ms\n", h2d_time + gpu_exec_time + d2h_time);

        // Destroy events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Free Memory
        free(h_M);
        free(h_N);
        free(h_P_cpu);
        free(h_P_gpu);
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
    }

    return 0;
}
