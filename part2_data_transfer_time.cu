#include <stdio.h>
#include <cuda.h>

// Function to measure Host → Device transfer time
float CPUtoGPUTime(float* d_ptr, float* h_ptr, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// Function to measure Device → Host transfer time
float GPUtoCPUTime(float* h_ptr, float* d_ptr, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);  // Fixed order
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int floatSize = sizeof(float);

    // Iterating through all the matrix sizes
    for (int i = 0; i < numOfSizes; i++) {
        int currentSize = matrixSizes[i];
        int bytes = currentSize * currentSize * floatSize;

        // Allocating memory on CPU
        float *h_M = (float*)malloc(bytes);
        float *h_N = (float*)malloc(bytes);

        // Filling in the matrices with random values
        for (int k = 0; k < currentSize * currentSize; k++) {
            h_M[k] = (float)(rand() % 5);  // Random values between 0-5
            h_N[k] = (float)(rand() % 5);
        }

        // Allocating memory on GPU
        float *d_M, *d_N;
        cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_N, bytes);

        // Measuring CPU → GPU transfer time
        float CPUtoGPUTime_MatrixM = CPUtoGPUTime(d_M, h_M, bytes);
        float CPUtoGPUTime_MatrixN = CPUtoGPUTime(d_N, h_N, bytes);
        float timeForCPUtoGPU = CPUtoGPUTime_MatrixM + CPUtoGPUTime_MatrixN;

        // Measuring GPU → CPU transfer time
        float GPUtoCPUTime_MatrixM = GPUtoCPUTime(h_M, d_M, bytes);
        float GPUtoCPUTime_MatrixN = GPUtoCPUTime(h_N, d_N, bytes);
        float timeForGPUtoCPU = GPUtoCPUTime_MatrixM + GPUtoCPUTime_MatrixN;

        // Print results
        printf("Matrix Size: %d x %d\n", currentSize, currentSize);
        printf("Host → Device Transfer Time: %.3f ms\n", timeForCPUtoGPU);
        printf("Device → Host Transfer Time: %.3f ms\n\n", timeForGPUtoCPU);

        // Free memory
        free(h_M);
        free(h_N);
        cudaFree(d_M);
        cudaFree(d_N);
    }

    return 0;
}