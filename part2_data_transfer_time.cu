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

// GPU Kernel for Matrix Multiplication (Single-threaded)
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

bool compareMatrices(float* A, float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // Array of matrix sizes
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int floatSize = sizeof(float);

    int blockSizes[] = {2, 4, 8, 16, 32};  // Different block widths
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

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

        // Copy input matrices from Host to Device
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

        printf("\nMatrix Size: %d x %d\n", N, N);

        for (int b = 0; b < numBlockSizes; b++) {
            int block_width = blockSizes[b];

            // Define grid and block dimensions
            dim3 threadsPerBlock(block_width, block_width);
            dim3 numBlocks((N + block_width - 1) / block_width, 
                           (N + block_width - 1) / block_width);

            // CUDA events for timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // **Measure GPU Execution Time**
            float gpu_exec_time;
            cudaEventRecord(start);
            MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_P, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_exec_time, start, stop);

            // Copy result back from GPU to CPU
            cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

            // **Print Execution Time**
            printf("Block Width: %d, GPU Execution Time: %.3f ms\n", block_width, gpu_exec_time);

            if (N <= 1024) {
                float cpu_exec_time;
                cudaEventRecord(start);
                matrixMulCPU(h_P_cpu, h_M, h_N, N);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&cpu_exec_time, start, stop);

                printf("CPU Execution Time: %.3f ms\n", cpu_exec_time);

                // **Compare CPU and GPU Results**
                bool isCorrect = compareMatrices(h_P_cpu, h_P_gpu, N * N, 1e-5);
                if (isCorrect) {
                    printf("Test PASSED \n");
                } else {
                    printf("Test FAILED \n");
                }
            }
             // Destroy events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

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