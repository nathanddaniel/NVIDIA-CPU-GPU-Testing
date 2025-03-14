#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

//use this for now, debuggin values later
#define MAX_BLOCK_WIDTH 32 

//tolerance for CPU/GPU comparison
#define tolerance 1e-5       

//single thread, single block matrix multiplication GPU kernel
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

//multi-threaded GPU Kernel
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

//CPU Matrix Multiplication function
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

    printf("\nData Transfer Time (Host to/from Device)\n");
    for (int size : sizes) {
        int matrixSize = size * size;
        float *hostM, *hostN, *deviceM, *deviceN;

        //allocating memory
        hostM = (float*)malloc(matrixSize * sizeof(float));
        hostN = (float*)malloc(matrixSize * sizeof(float));
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));

        //initializing matrices
        for (int i = 0; i < matrixSize; i++) {
            hostM[i] = (float)(rand() % 5);
            hostN[i] = (float)(rand() % 5);
        }

        //timing events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //Host (CPU) to Device (GPU) time transfer
        float timeH2D;
        cudaEventRecord(start, 0);
        cudaMemcpy(deviceM, hostM, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, hostN, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeH2D, start, stop);
        printf("Matrix Size: %d x %d --- Host to Device: %.3f ms\n", size, size, timeH2D);

        //Device (CPU) to Host (GPU) time ransfer
        float timeD2H;
        cudaEventRecord(start, 0);
        cudaMemcpy(hostM, deviceM, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostN, deviceN, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeD2H, start, stop);
        printf("Matrix Size: %d x %d --- Device to Host: %.3f ms\n", size, size, timeD2H);

        //freeing memory
        free(hostM);
        free(hostN);
        cudaFree(deviceM);
        cudaFree(deviceN);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\nSingle-threaded GPU vs. CPU\n");
    for (int size : {256, 512, 1024}) {
        int matrixSize = size * size;

        //allocating CPU Memory for matrices M and N
        float *hostM, *hostN, *cpuHostP, *kernelHostP;
        hostM = (float*)malloc(matrixSize * sizeof(float));
        hostN = (float*)malloc(matrixSize * sizeof(float));
        cpuHostP = (float*)malloc(matrixSize * sizeof(float));  
        kernelHostP = (float*)malloc(matrixSize * sizeof(float));

        //allocate GPU Memory
        float *deviceM, *deviceN, *deviceP;
        cudaMalloc(&deviceM, matrixSize * sizeof(float));
        cudaMalloc(&deviceN, matrixSize * sizeof(float));
        cudaMalloc(&deviceP, matrixSize * sizeof(float));

        //filling in values for the matrices M and N (inputs for the calcualtions)
        for (int i = 0; i < matrixSize; i++) {
            hostM[i] = (float)(rand() % 5);
            hostN[i] = (float)(rand() % 5);
        }

        //copying the matrices over to the device
        cudaMemcpy(deviceM, hostM, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceN, hostN, matrixSize * sizeof(float), cudaMemcpyHostToDevice);

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
        kernelMatMul_SingleThread<<<gridDim, blockDim>>>(deviceP, deviceM, deviceN, size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&totalGPUTime, start, stop);
        
        //printing the matrix size and its corresponding single threaded GPU calculation time
        printf("Matrix Size is: %d x %d --- Single-Threaded GPU Time is: %.3f ms\n", size, size, totalGPUTime);

        //copying the GPU Result to the CPU
        cudaMemcpy(kernelHostP, deviceP, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

        //host execution timing
        float totalCPUTime;
        cudaEventRecord(start, 0);
        cpuMatMul(cpuHostP, hostM, hostN, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&totalCPUTime, start, stop);

        printf("Matrix Size is: %d x %d --- CPU Computation Time is: %.3f ms\n", size, size, totalCPUTime);

        //comparing the CPU and GPU Results
        bool passed = true;
        for (int i = 0; i < matrixSize; i++) {
            if (fabs(kernelHostP[i] - cpuHostP[i]) > tolerance) {
                printf("There's incorrect values at these indexes; %d: CPU = %.6f, GPU = %.6f\n", i, cpuHostP[i], kernelHostP[i]);
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
            kernelMatMul_MultiThread<<<gridDim, blockDim>>>(deviceP, deviceP, deviceP, size);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&multiThreadTime, start, stop);
            printf("Matrix Size is: %d x %d --- Block Width is: %d --- Multi-Threaded GPU Time is: %.3f ms\n",
                   size, size, blockWidth, multiThreadTime);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaFree(deviceP);
    }

    return 0;
}