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
    // Since this is a single-threaded GPU kernel, we manually iterate over the matrix
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

int main() {
    // Array of matrix sizes
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int floatSize = sizeof(float);

    for (int i = 0; i < numOfSizes; i++) {
        int N = matrixSizes[i];
        int bytes = N * N * floatSize;

        // Allocate memory on CPU
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);

        // Initialize matrices with random values
        for (int j = 0; j < N * N; j++) {
            h_M[j] = (float)(rand() % 5);
            h_N[j] = (float)(rand() % 5);
        }

        // Allocate memory on GPU
        float* d_M, *d_N;
        cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_N, bytes);

        // CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // **Host to Device Transfer**
        float h2d_time;
        cudaEventRecord(start);
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&h2d_time, start, stop);

        // **Print Data Transfer Time**
        printf("\nMatrix Size: %d x %d\n", N, N);
        printf("Host → Device Transfer Time: %.3f ms\n", h2d_time);

        // **Only Execute Matrix Multiplication for Certain Sizes**
        if (N <= 1024) {  // Perform computation only for 256x256, 512x512, 1024x1024
            float* h_P_cpu = (float*)malloc(bytes);
            float* h_P_gpu = (float*)malloc(bytes);
            float* d_P;
            cudaMalloc(&d_P, bytes);

            // **GPU Execution**
            float gpu_exec_time;
            cudaEventRecord(start);
            MatrixMulKernel<<<1, 1>>>(d_M, d_N, d_P, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_exec_time, start, stop);

            // **Device to Host Transfer**
            float d2h_time;
            cudaEventRecord(start);
            cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&d2h_time, start, stop);

            // **CPU Execution Time**
            float cpu_exec_time = 0.0;
            cudaEventRecord(start, 0);
            cudaDeviceSynchronize();
            matrixMulCPU(h_P_cpu, h_M, h_N, N);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&cpu_exec_time, start, stop);

            // **Print Execution Times**
            printf("GPU Execution Time: %.3f ms\n", gpu_exec_time);
            printf("Device → Host Transfer Time: %.3f ms\n", d2h_time);
            printf("CPU Execution Time: %.3f ms\n", cpu_exec_time);

            // Free extra memory only when multiplication was performed
            free(h_P_cpu);
            free(h_P_gpu);
            cudaFree(d_P);
        } 
        
        else {
            printf("Skipping matrix multiplication for %d x %d matrix\n", N, N);
        }

        // Free Memory
        free(h_M);
        free(h_N);
        cudaFree(d_M);
        cudaFree(d_N);
    }

    return 0;
}