#include "cuda_runtime.h" //CUDA API functions
#include <string.h> 
#include <stdio.h> //used for printing stuff for like name of CUDA able GPUs


int main() {

    int nd; //number of avialble CUDA GPUs

    //the function cudaGetDeviceCount() returns the number of CUDA capable GPUs
    nd = cudaGetDeviceCount(&nd);

    //printing out the number of CUDA capable GPUs available
    printf("Number of CUDA capable GPUs that are available is: %d\n", nd);

    //now we're going to loop through each CUDA GPU 
    for (int i = 0; i < nd; i++){

        /* we're creating a variable dp of type cudaDeviceProp which
        is a struct that contains the properties of the CUDA GPU
        */
        cudaDeviceProp dp;

        //filling in the structure with the properties of the GPU
        cudaDeviceProperties(&dp, i);

        //printing out the GPU info
        printf("Device %d --> Max Threads per SM: %d, Warp Size: %d\n", i, dp.maxThreadsPerMultiProcessor, dp.warpSize);

        //printing the GPU Model Name
        printf("GPU Name: %s\n", dp.name);
    }

    return 0;
}

for (int d = 0; d < nd; d++){
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, d);
    printf("Device %d --> Property 1 is %d, Property 2 is %d\n", d, dp.maxThreadsPerMultiProcessor, dp.warpSize);
    strcpy(name, dp.name);
    printf("GPU Name: %s\n", name);
}
