#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>

// CUDA Kernel: Increments each element in g_data by inc_value
__global__ void increment_kernel(int *g_data, int inc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

// Function to check if output is correct
int correct_output(int *data, const int n, const int x) {
    for(int i = 0; i < n; i++)
        if(data[i] != x)
            return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    cudaDeviceProp deviceProps;
    
    // Get device name
    cudaGetDeviceProperties(&deviceProps, 0);
    printf("CUDA device [%s]\n", deviceProps.name);

    int n = 16 * 1024 * 1024;  // Array size
    int nbytes = n * sizeof(int);
    int value = 26;  // Increment value

    // Allocate host memory
    int *a = 0;
    cudaMallocHost((void**)&a, nbytes);
    
    // Allocate device memory
    int *d_a = 0;
    cudaMalloc((void**)&d_a, nbytes);
    cudaMemset(d_a, 255, nbytes);  // Initialize device memory to 255

    // Set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);

    // Create CUDA event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    float gpu_time = 0.0f;

    // Asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);

    // Have CPU do some work while waiting for GPU to finish
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;  // Indicates that the CPU is running asynchronously while GPU is executing
    }

    cudaEventSynchronize(stop);  // Ensure stop event is updated
    cudaEventElapsedTime(&gpu_time, start, stop);  // Calculate elapsed time

    // Print GPU execution time
    printf("Time spent executing by the GPU: %.2f ms\n", gpu_time);
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // Check output correctness
    bool bFinalResults = (bool)correct_output(a, n, value);
    printf("Output correctness: %s\n", bFinalResults ? "PASS" : "FAIL");

    // Release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(a);
    cudaFree(d_a);
    cudaDeviceReset();

    return 0;
}
