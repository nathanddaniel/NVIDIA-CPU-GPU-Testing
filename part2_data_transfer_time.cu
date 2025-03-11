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

bool compareMatrices(float* A, float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;  // Matrices do not match
        }
    }
    return true;  // Matrices match
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

        float* h_P_cpu = (float*)malloc(bytes);
        float* h_P_gpu = (float*)malloc(bytes);
        float* d_P;
        cudaMalloc(&d_P, bytes);

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
        printf("Host to Device Transfer Time: %.3f ms\n", h2d_time);

        // **Device to Host Transfer**
        float d2h_time;
        cudaEventRecord(start);
        cudaMemcpy(h_M, d_M, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_N, d_N, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&d2h_time, start, stop);

        printf("Device to Host Transfer Time: %.3f ms\n", d2h_time);

        // **Only Execute Matrix Multiplication for Certain Sizes**
        if (N <= 1024) {  // Perform computation only for 256x256, 512x512, 1024x1024
            // **GPU Execution**
            float gpu_exec_time;
            cudaEventRecord(start);
            MatrixMulKernel<<<1, 1>>>(d_M, d_N, d_P, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpu_exec_time, start, stop);

            // Copy result back from GPU to CPU
            cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

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
            printf("CPU Execution Time: %.3f ms\n", cpu_exec_time);

            // **Compare Results**
            bool isCorrect = compareMatrices(h_P_cpu, h_P_gpu, N * N, 1e-5);

            if (isCorrect) {
                printf("Test PASSED \n");
            } else {
                printf("Test FAILED \n");
            }
        } 
        
        else {
            printf("Skipping matrix multiplication for %d x %d matrix\n", N, N);
        }

        // Free Memory
        free(h_P_cpu);
        free(h_P_gpu);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        cudaFree(d_M);
        cudaFree(d_N);
    }

    return 0;
}