#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

#define MAX_BLOCK_WIDTH 32  // Maximum block width to test
#define EPSILON 1e-5        // Tolerance for result comparison

// **Single-Threaded GPU Kernel**
__global__ void kernelMatMul_SingleThread(float* P, float* M, float* N, int size) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
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
}

// **Multi-Threaded GPU Kernel**
__global__ void kernelMatMul_MultiThread(float* P, float* M, float* N, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += M[row * size + i] * N[i * size + col];
        }
        P[row * size + col] = sum;
    }
}

// **CPU Matrix Multiplication**
void cpuMatMul(float* P, float* M, float* N, int size) {
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
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    std::vector<int> blockWidths = {2, 4, 8, 16, 32};

    printf("\n========= Data Transfer Time (Host to Device) =========\n");
    for (int size : sizes) {
        int matrixSize = size * size;
        float *hostM, *hostN, *deviceM, *deviceN;

        // Allocate memory
        hostM = (float*)malloc(matrixSize * sizeof(float));
        hostN = (float*)malloc(matrixSize * sizeof(float));
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));

        // Initialize matrices
        for (int i = 0; i < matrixSize; i++) {
            hostM[i] = static_cast<float>(rand()) / RAND_MAX;
            hostN[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Timing events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // **Host to Device Transfer**
        float timeH2D;
        cudaEventRecord(start, 0);
        cudaMemcpy(deviceM, hostM, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, hostN, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeH2D, start, stop);
        printf("Matrix Size: %d x %d | Host to Device: %.3f ms\n", size, size, timeH2D);

        // **Device to Host Transfer**
        float timeD2H;
        cudaEventRecord(start, 0);
        cudaMemcpy(hostM, deviceM, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostN, deviceN, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeD2H, start, stop);
        printf("Matrix Size: %d x %d | Device to Host: %.3f ms\n", size, size, timeD2H);

        // Cleanup
        free(hostM);
        free(hostN);
        cudaFree(deviceM);
        cudaFree(deviceN);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n========= Single-Threaded GPU vs. CPU Computation =========\n");
    for (int size : {256, 512, 1024}) {
        int matrixSize = size * size;

        // **Allocate Host Memory for Input & Output Matrices**
        float *hostM, *hostN, *cpuHostP, *kernelHostP;
        hostM = (float*)malloc(matrixSize * sizeof(float));
        hostN = (float*)malloc(matrixSize * sizeof(float));
        cpuHostP = (float*)malloc(matrixSize * sizeof(float));  // CPU output
        kernelHostP = (float*)malloc(matrixSize * sizeof(float)); // GPU output

        // **Allocate Device Memory**
        float *deviceM, *deviceN, *deviceP;
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));
        cudaMalloc(&deviceP, matrixSize * sizeof(float));

        // **Initialize Input Matrices**
        for (int i = 0; i < matrixSize; i++) {
            hostM[i] = static_cast<float>(rand()) / RAND_MAX;
            hostN[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // **Copy Matrices to Device**
        cudaMemcpy(deviceM, hostM, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, hostN, matrixSize * sizeof(float), cudaMemcpyHostToDevice);

        // **Single-Threaded GPU Execution**
        dim3 gridDim(1, 1);
        dim3 blockDim(1, 1);

        float gpuTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        kernelMatMul_SingleThread<<<gridDim, blockDim>>>(deviceP, deviceM, deviceN, size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        
        printf("Matrix Size: %d x %d | Single-Threaded GPU: %.3f ms\n", size, size, gpuTime);

        // **Copy GPU Result to Host**
        cudaMemcpy(kernelHostP, deviceP, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

        // **CPU Execution Timing**
        float cpuTime;
        cudaEventRecord(start, 0);
        cpuMatMul(cpuHostP, hostM, hostN, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpuTime, start, stop);

        printf("Matrix Size: %d x %d | CPU Computation: %.3f ms\n", size, size, cpuTime);

        // **Compare CPU and GPU Results**
        bool isCorrect = true;
        for (int i = 0; i < matrixSize; i++) {
            if (fabs(kernelHostP[i] - cpuHostP[i]) > EPSILON) {
                printf("Mismatch at index %d: CPU = %.6f, GPU = %.6f\n", i, cpuHostP[i], kernelHostP[i]);
                isCorrect = false;
                break;  // Stop checking after first mismatch (optional)
            }
        }

        // **Print Final Verification Result**
        if (isCorrect) {
            printf("Test PASSED\n");
        } else {
            printf("Test FAILED\n");
        }

        // **Free Memory**
        free(hostM);
        free(hostN);
        free(cpuHostP);
        free(kernelHostP);
        cudaFree(deviceM);
        cudaFree(deviceN);
        cudaFree(deviceP);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n========= Multi-Threaded GPU Computation =========\n");
    for (int size : sizes) {
        int matrixSize = size * size;
        float *deviceP;
        cudaMalloc(&deviceP, matrixSize * sizeof(float));

        for (int blockWidth : blockWidths) {
            dim3 gridDim((size + blockWidth - 1) / blockWidth, (size + blockWidth - 1) / blockWidth);
            dim3 blockDim(blockWidth, blockWidth);

            // **Multi-Threaded GPU Execution**
            float multiThreadTime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            kernelMatMul_MultiThread<<<gridDim, blockDim>>>(deviceP, deviceP, deviceP, size);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&multiThreadTime, start, stop);
            printf("Matrix Size: %d x %d | Block Width: %d | Multi-Threaded GPU: %.3f ms\n",
                   size, size, blockWidth, multiThreadTime);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaFree(deviceP);
    }

    return 0;
}