#include <stdio.h>
#include <cuda.h>

//find time it takes to transfer the two input matrices from host to device

/* 

matrix sizes: 

256 x 256
512 x 512
1024 x 1024
2048 x 2048
4096 x 4096

*/

//plot data transfer time vs matrix size


//now transfer the matrices back from device (GPU) to the host (CPU)


//function that measures time it takes between GPU and CPU as well 
float CPUtoGPUTime(float* d_ptr, float* h_ptr, int size) {
    
    cudaEvent_t start;
    cudaEvent_t stop;

    //creating event that records time before the data transfer
    cudaEventCreate(&start);

    //creating event that records time after the data transfer
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

float GPUtoCPUTime(float* h_ptr, float* d_ptr, int size) {
    cudaEvent_t start;
    cudaEvent_t stop;

    //creating event that records time before the data transfer
    cudaEventCreate(&start);

    //creating event that records time after the data transfer
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyDeviceToHost);
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

    //finding number of items in the array
    int numOfSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);

    int floatSize = sizeof(float);

    //iteraing through all the matrix sizes we have 
    for (int i = 0; i < numOfSizes; i++) {

        //the current size to implement
        int currentSize = matrixSizes[i];

        //number of bytes for the current size
        int bytes = (currentSize) * (currentSize) * (floatSize);



        //allocating memory on the CPU to store matrices M and N on
        float *h_M = (float*)malloc(bytes);
        float *h_N = (float*)malloc(bytes);

        //filling in the matrices with values (random ones tho)
        for (int k = 0; k < currentSize * currentSize; k++){
            //generating rando values between 0 and 5, small values to make it easeier to debug for now
            h_M[k] = (float)(rand() % 5);
            h_N[k] = (float)(rand() % 5);
        }



        //allocating memeory on the GPU to store matrices M and N on 
        float *d_M;
        float *d_N;

        cudaMalloc(&d_M, bytes);
        cudaMalloc(&d_N, bytes);

        //measuring CPU to GPU copying time for each of the matrices M and N
        float CPUtoGPUTime_MatrixM = CPUtoGPUTime(d_M, h_M, bytes);
        float CPUtoGPUTime_MatrixN = CPUtoGPUTime(d_N, h_N, bytes);
        float timeForCPUtoGPU = CPUtoGPUTime_MatrixM + CPUtoGPUTime_MatrixN;

        //measuring GPU to CPU copying time for each of the matrices M and N
        float GPUtoCPUTime_MatrixM = GPUtoCPUTime(h_M, d_M, bytes);
        float GPUtoCPUTime_MatrixN = GPUtoCPUTime(h_N, d_N, bytes);
        float timeForGPUtoCPU = GPUtoCPUTime_MatrixM + GPUtoCPUTime_MatrixN;

        // Print results
        printf("Matrix Size: %d x %d\n", currentSize, currentSize);
        printf("Host → Device Transfer Time: %.3f ms\n", timeForCPUtoGPU);
        printf("Device → Host Transfer Time: %.3f ms\n\n", timeForGPUtoCPU);

        free(h_M);
        free(h_N);
        cudaFree(d_M);
        cudaFree(d_N);
    }

    return 0;

}