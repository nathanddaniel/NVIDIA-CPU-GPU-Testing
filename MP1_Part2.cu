#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

//tolerance for CPU/GPU comparison
#define tolerance 1e-5       

//single thread, single block matrix multiplication GPU kernel
__global__ void singleThreadMatrixMul(float* P, float* M, float* N, int size) {
    //single thread if condition 
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

//multi-threaded GPU Kernel
__global__ void multiThreadMatrixMul(float* P, float* M, float* N, int size) {
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

//CPU Matrix Multiplication function
void CPUMatMul(float* P, float* M, float* N, int size) {
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

    printf("\nData Transfer Time (Host to/from Device)\n");
    for (int size : sizes) {
        int matrixSize = size * size;
        float *h_M, *h_N, *deviceM, *deviceN;

        //allocating memory
        h_M = (float*)malloc(matrixSize * sizeof(float));
        h_N = (float*)malloc(matrixSize * sizeof(float));
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));

        //initializing matrices
        for (int i = 0; i < matrixSize; i++) {
            h_M[i] = (float)(rand() % 5);
            h_N[i] = (float)(rand() % 5);
        }

        //timing events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //Host (CPU) to Device (GPU) time transfer
        float timeH2D;
        cudaEventRecord(start, 0);
        cudaMemcpy(deviceM, h_M, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, h_N, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeH2D, start, stop);
        printf("Matrix Size: %d x %d --- Host to Device: %.3f ms\n", size, size, timeH2D);

        //Device (CPU) to Host (GPU) time ransfer
        float timeD2H;
        cudaEventRecord(start, 0);
        cudaMemcpy(h_M, deviceM, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_N, deviceN, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeD2H, start, stop);
        printf("Matrix Size: %d x %d --- Device to Host: %.3f ms\n", size, size, timeD2H);

        //freeing memory
        free(h_M);
        free(h_N);
        cudaFree(deviceM);
        cudaFree(deviceN);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\nSingle-threaded GPU vs. CPU\n");
    for (int size : {256, 512, 1024}) {
        int matrixSize = size * size;

        //allocating CPU Memory for matrices M and N
        float *h_M, *h_N, *c_HP, *k_HP;
        h_M = (float*)malloc(matrixSize * sizeof(float));
        h_N = (float*)malloc(matrixSize * sizeof(float));
        c_HP = (float*)malloc(matrixSize * sizeof(float));  
        k_HP = (float*)malloc(matrixSize * sizeof(float));

        //allocate GPU Memory
        float *deviceM, *deviceN, *deviceP;
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));
        cudaMalloc(&deviceP, matrixSize * sizeof(float));

        //filling in values for the matrices M and N (inputs for the calcualtions)
        for (int i = 0; i < matrixSize; i++) {
            h_M[i] = (float)(rand() % 5);
            h_N[i] = (float)(rand() % 5);
        }

        //copying the matrices over to the device
        cudaMemcpy(deviceM, h_M, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, h_N, matrixSize * sizeof(float), cudaMemcpyHostToDevice);

        //specifying the number of blocks in the grid (1 for 2.2)
        dim3 gridDim(1, 1);

        //specifying the number of threads in a block (1 for 2.2 where we just want 1 block and 1 thread within that one block)
        dim3 blockDim(1, 1);

        //cuda Time events
        float totalGPUTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        singleThreadMatrixMul<<<gridDim, blockDim>>>(deviceP, deviceM, deviceN, size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&totalGPUTime, start, stop);
        
        //printing the matrix size and its corresponding single threaded GPU calculation time
        printf("Matrix Size is: %d x %d --- Single-Threaded GPU Time is: %.3f ms\n", size, size, totalGPUTime);

        //copying the GPU Result to the CPU
        cudaMemcpy(k_HP, deviceP, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

        //host execution timing
        float totalCPUTime;
        cudaEventRecord(start, 0);
        CPUMatMul(c_HP, h_M, h_N, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&totalCPUTime, start, stop);

        printf("Matrix Size is: %d x %d --- CPU Computation Time is: %.3f ms\n", size, size, totalCPUTime);

        //comparing the CPU and GPU Results
        bool passed = true;
        for (int i = 0; i < matrixSize; i++) {
            if (fabs(k_HP[i] - c_HP[i]) > tolerance) {
                printf("There's incorrect values at these indexes; %d: CPU = %.6f, GPU = %.6f\n", i, c_HP[i], k_HP[i]);
                passed = false;
                break;  
            }
        }

        //printing whether the host and device computation results are the same
        if (passed) {
            printf("Test PASSED\n");
        } else {
            printf("Test FAILED\n");
        }

        //freeing up memory
        free(h_M);
        free(h_N);
        free(c_HP);
        free(k_HP);
        cudaFree(deviceM);
        cudaFree(deviceN);
        cudaFree(deviceP);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n multi threaded device computation \n");
    for (int size : sizes) {
        int matrixSize = size * size;
        float *deviceP;
        cudaMalloc(&deviceP, matrixSize * sizeof(float));

        for (int blockWidth : blockWidths) {
            dim3 gridDim((size + blockWidth - 1) / blockWidth, (size + blockWidth - 1) / blockWidth);
            dim3 blockDim(blockWidth, blockWidth);

            //multi-threaded GPU Execution
            float multiThreadTime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            multiThreadMatrixMul<<<gridDim, blockDim>>>(deviceP, deviceP, deviceP, size);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&multiThreadTime, start, stop);
            printf("Matrix Size is: %d x %d --- Block Width is: %d --- Multi-Threaded GPU Time is: %.3f ms\n", size, size, blockWidth, multiThreadTime);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaFree(deviceP);
    }

    return 0;
}