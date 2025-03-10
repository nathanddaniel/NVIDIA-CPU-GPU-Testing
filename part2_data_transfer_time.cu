#include <stdio.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <cmath>

// Matrix Sizes
#define BLOCK_WIDTH 32
#define MAT_DIM_1 256
#define MAT_DIM_2 512
#define MAT_DIM_3 1024

// GPU Kernel for Matrix Multiplication (Single-threaded)
__global__ void MatrixMulKernel(float* M, float* N, float* P, int size) {
    // Since this is a single-threaded GPU kernel, we manually iterate over the matrix
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            float sum = 0.0;
            for (int i = 0; i < size; i++) {
                sum += M[row * size + i] * N[i * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}

// CPU Matrix Multiplication
void matrixMulCPU(float* P, float* M, float* N, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            float sum = 0.0;
            for (int i = 0; i < size; i++) {
                sum += M[row * size + i] * N[i * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}

int main() {
    // Array of matrix sizes
    int matrixSizes[] = {MAT_DIM_1, MAT_DIM_2, MAT_DIM_3};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);

    for (int i = 0; i < numOfSizes; i++) {
        int N = matrixSizes[i];
        int bytes = N * N * sizeof(float);

        // Allocate memory on CPU
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P_cpu = (float*)malloc(bytes);
        float* h_P_gpu = (float*)malloc(bytes);

        // Initialize matrices with random values
        for (int j = 0; j < N * N; j++) {
            h_M[j] = static_cast<float>(rand()) / RAND_MAX;
            h_N[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory on GPU
        float* d_M, *d_N, *d_P;
        cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_N, bytes);
        cudaMalloc(&d_P, bytes);

        // Create CUDA event handles
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // **Host to Device Transfer Time**
        float h2d_time = 0.0;
        cudaEventRecord(start, 0);
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&h2d_time, start, stop);

        // **Kernel Launch Configuration (Single-Threaded Execution for Part 2)**
        dim3 gridDim(1, 1);  // One block
        dim3 blockDim(1, 1); // One thread

        // **GPU Execution Time**
        float gpu_exec_time = 0.0;
        cudaEventRecord(start, 0);
        cudaDeviceSynchronize();
        MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_exec_time, start, stop);

        // **Device to Host Transfer Time**
        float d2h_time = 0.0;
        cudaEventRecord(start, 0);
        cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
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

        // **Print Results**
        printf("\nMatrix Size: %d x %d\n", N, N);
        printf("Host to Device Transfer Time: %.3f ms\n", h2d_time);
        printf("GPU Execution Time: %.3f ms\n", gpu_exec_time);
        printf("Device to Host Transfer Time: %.3f ms\n", d2h_time);
        printf("CPU Execution Time: %.3f ms\n", cpu_exec_time);

        // **Validate GPU Output Against CPU**
        bool flag = true;
        for (int i = 0; i < N * N; i++) {
            if (fabs(h_P_gpu[i] - h_P_cpu[i]) > 1e-5) {
                flag = false;
                break;
            }
        }
        if (flag) {
            printf("Test PASSED\n");
        } else {
            printf("Test FAILED\n");
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