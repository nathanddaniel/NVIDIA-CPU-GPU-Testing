#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define MAT_DIM 512  // Change this manually to 256, 512, 1024, etc.
#define BLOCK_WIDTH 32
#define MAT_SIZE (MAT_DIM * MAT_DIM)

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
    // **Allocate memory for input/output matrices on host**
    float* hostM = (float*)malloc(MAT_SIZE * sizeof(float));
    float* hostN = (float*)malloc(MAT_SIZE * sizeof(float));
    float* kernelHostP = (float*)malloc(MAT_SIZE * sizeof(float));
    float* cpuHostP = (float*)malloc(MAT_SIZE * sizeof(float));

    // **Allocate memory on device**
    float *deviceM, *deviceN, *deviceP;
    cudaMalloc(&deviceM, MAT_SIZE * sizeof(float));
    cudaMalloc(&deviceN, MAT_SIZE * sizeof(float));
    cudaMalloc(&deviceP, MAT_SIZE * sizeof(float));

    // **Initialize input matrices**
    for (int i = 0; i < MAT_SIZE; i++) {
        hostM[i] = static_cast<float>(rand()) / RAND_MAX;
        hostN[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // **Copy matrices to device**
    cudaMemcpy(deviceM, hostM, MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, hostN, MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // **Single-Thread GPU Execution**
    dim3 gridDimSingle(1, 1);
    dim3 blockDimSingle(1, 1);

    float singleThreadTime = 0.0;
    cudaEventRecord(start, 0);
    kernelMatMul_SingleThread<<<gridDimSingle, blockDimSingle>>>(deviceP, deviceM, deviceN, MAT_DIM);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&singleThreadTime, start, stop);
    printf("Single-Thread GPU Computation Time: %.3f ms\n", singleThreadTime);

    // // **Multi-Threaded GPU Execution**
    // dim3 gridDimMulti((MAT_DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 
    //                   (MAT_DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
    // dim3 blockDimMulti(BLOCK_WIDTH, BLOCK_WIDTH);

    // float multiThreadTime = 0.0;
    // cudaEventRecord(start, 0);
    // kernelMatMul_MultiThread<<<gridDimMulti, blockDimMulti>>>(deviceP, deviceM, deviceN, MAT_DIM);
    // cudaDeviceSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&multiThreadTime, start, stop);
    // printf("Multi-Threaded GPU Computation Time: %.3f ms\n", multiThreadTime);

    // **CPU Computation**
    float cpuTime = 0.0;
    cudaEventRecord(start, 0);
    cpuMatMul(cpuHostP, hostM, hostN, MAT_DIM);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpuTime, start, stop);
    printf("CPU Computation Time: %.3f ms\n", cpuTime);

    // **Copy results from device to host**
    cudaMemcpy(kernelHostP, deviceP, MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // **Compare CPU and GPU Results (Threshold = 1e-5)**
    bool isCorrect = true;
    const float epsilon = 1e-5;
    for (int i = 0; i < MAT_SIZE; i++) {
        if (fabs(kernelHostP[i] - cpuHostP[i]) > epsilon) {
            printf("Mismatch at index %d: CPU = %.6f, GPU = %.6f\n", 
                i, cpuHostP[i], kernelHostP[i]);
            isCorrect = false;
            break;  // Stop checking after the first error (optional)
        }
    }
    // **Print Final Result**
    if (isCorrect) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // **Free memory**
    free(hostM);
    free(hostN);
    free(kernelHostP);
    free(cpuHostP);
    cudaFree(deviceM);
    cudaFree(deviceN);
    cudaFree(deviceP);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}