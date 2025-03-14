#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define BLOCK_WIDTH 32
#define MAT_DIM 500
#define MAT_SIZE (MAT_DIM * MAT_DIM)

// **GPU Kernel for Matrix Multiplication**
__global__ void kernelMatMul(float* P, float* M, float* N, int size) {
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
    // **Allocate memory for input/output matrices on host (CPU)**
    float* hostM = (float*)malloc(MAT_SIZE * sizeof(float));
    float* hostN = (float*)malloc(MAT_SIZE * sizeof(float));
    float* kernelHostP = (float*)malloc(MAT_SIZE * sizeof(float));
    float* cpuHostP = (float*)malloc(MAT_SIZE * sizeof(float));

    // **Allocate memory on device (GPU)**
    float *deviceM, *deviceN, *deviceP;
    cudaMalloc(&deviceM, MAT_SIZE * sizeof(float));
    cudaMalloc(&deviceN, MAT_SIZE * sizeof(float));
    cudaMalloc(&deviceP, MAT_SIZE * sizeof(float));

    // **Initialize input matrices**
    for (int i = 0; i < MAT_SIZE; i++) {
        hostM[i] = static_cast<float>(rand()) / RAND_MAX;
        hostN[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // **Create CUDA event handles for timing**
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // **Memcpy time transfer (Host to Device)**
    float time1 = 0.0;
    cudaEventRecord(start, 0);
    cudaMemcpy(deviceM, hostM, MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, hostN, MAT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    printf("Data transfer time - Host to Device: %.3f ms\n", time1);

    // **Memcpy time transfer (Device to Host)**
    float time2 = 0.0;
    cudaEventRecord(start, 0);
    cudaMemcpy(hostM, deviceM, MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostN, deviceN, MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time2, start, stop);
    printf("Data transfer time - Device to Host: %.3f ms\n", time2);

    // **Kernel Execution Configuration**
    // **(Using 1 block, 1 thread per block)**
    dim3 gridDim(1, 1);
    dim3 blockDim(1, 1);

    // **Using multiple blocks and threads**
    dim3 gridDimOptimized((MAT_DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 
                          (MAT_DIM + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
    dim3 blockDimOptimized(BLOCK_WIDTH, BLOCK_WIDTH);

    // **Record GPU matrix multiplication time (Parallel Execution)**
    float time3 = 0.0;
    cudaEventRecord(start, 0);
    kernelMatMul<<<gridDimOptimized, blockDimOptimized>>>(deviceP, deviceM, deviceN, MAT_DIM);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time3, start, stop);
    printf("GPU Computation Time: %.3f ms\n", time3);

    // **Copy results from device to host**
    cudaMemcpy(kernelHostP, deviceP, MAT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // **Record CPU matrix multiplication time**
    float time4 = 0.0;
    cudaEventRecord(start, 0);
    cpuMatMul(cpuHostP, hostM, hostN, MAT_DIM);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time4, start, stop);
    printf("CPU Computation Time: %.3f ms\n", time4);

    // **Print total GPU time (including data transfer)**
    printf("Total time for GPU: %.3f ms\n", time1 + time2 + time3);

    // **Compare CPU and GPU Results (Threshold = 1e-10)**
    bool flag = true;
    for (int i = 0; i < MAT_SIZE; i++) {
        if (fabs(kernelHostP[i] - cpuHostP[i]) > 1e-10) {
            flag = false;
            break;
        }
    }

    // **Output results**
    if (flag) {
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

    return 0;
}