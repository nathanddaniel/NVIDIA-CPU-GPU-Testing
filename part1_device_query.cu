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
        //should I do CUDA Core per SM count for my GPU model?

        //printing the amount of Global Memory
        printf("Amount of Global Memory is: %d \n", dp.totalGlobalMem);

        
        //printing the amount of Constant Memory
        printf("Amount of Global Memory is: %d \n", dp.totalConstMem);

        //printing the GPU Model Name
        printf("GPU Name: %s\n", dp.name);
    }

    return 0;
}
