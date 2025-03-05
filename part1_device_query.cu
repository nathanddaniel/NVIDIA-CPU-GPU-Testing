#include "cuda_runtime.h" //CUDA API functions
#include <string.h> 
#include <stdio.h> //used for printing stuff for like name of CUDA able GPUs


int main() {

    int nd; //number of avialble CUDA GPUs

    //the function cudaGetDeviceCount() returns the number of CUDA capable GPUs
    cudaGetDeviceCount(&nd);

    //printing out the number of CUDA capable GPUs available
    printf("Number of CUDA capable GPUs that are available is: %d\n", nd);

    //now we're going to loop through each CUDA GPU 
    for (int i = 0; i < nd; i++){

        /* we're creating a variable dp of type cudaDeviceProp which
        is a struct that contains the properties of the CUDA GPU
        */
        cudaDeviceProp dp;

        //filling in the structure with the properties of the GPU
        cudaGetDeviceProperties(&dp, i);

        //printing out the GPU info
        printf("Device %d --> Max Threads per SM: %d, Warp Size: %d\n", i, dp.maxThreadsPerMultiProcessor, dp.warpSize);

        //printing out the clock info
        printf("Clock Rate is: %d kHz\n", dp.clockRate);

        //printing out the number of SM's
        printf("Number of Streaming Multiprocessors is: %d\n", dp.multiProcessorCount);

        //printing out the number of cores
        //formula CUDA cores = (# of SM)*(CUDA core per SM)
        //for the RTX 3060 Ti theres 128 CUDA cores/SM
        int numCudaCores = dp.multiProcessorCount * 128; // 128 cores per SM for Ampere
        printf("Number of CUDA Cores: %d\n", numCudaCores);

        //printing the amount of Global Memory
        printf("Amount of Global Memory is: %llu bytes \n", dp.totalGlobalMem);

        
        //printing the amount of Constant Memory
        printf("Amount of Constant Memory is: %llu bytes \n", dp.totalConstMem);

        //printing the GPU Model Name
        printf("GPU Name: %s\n", dp.name);

        //printing the shared memory for each block
        printf("Shared Memory Per Block: %d \n", dp.sharedMemPerBlock);

        //printing the number of registers per block
        printf("Number of registers/block: %d \n", dp.regsPerBlock);

        //printing the maximum number of threads per block
        printf("Max number of threads per block: %d \n", dp.maxThreadsPerBlock);

        // Printing the maximum size of each dimension of a block (x, y, z)
        printf("Max size of each dimension of a block: x = %d, y = %d, z = %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);

        // Printing the maximum size of each dimension of a grid (x, y, z)
        printf("Max size of each dimension of a grid: x = %d, y = %d, z = %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
    }

    return 0;
}
