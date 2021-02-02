#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <numeric>
#include <string>
#include <functional>

using namespace std;

cudaError_t customCudaError(cudaError_t result)
{
	if (result != cudaSuccess)
    	{
        	fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	        assert(result == cudaSuccess);
    	}
   	return result;
}

__global__
void hashFunction(unsigned int *rainbow, unsigned int columnCount)
{
	int th = blockIdx.x * blockDim.x + threadIdx.x;
    	if (th % columnCount == 0)
	{
		unsigned long long copy = rainbow[th];
		copy = ((copy >> 8) ^ copy) * 0x45d9f3b;
		copy = ((copy >> 8) ^ copy) * 0x45d9f3b;
		copy = (copy >> 8) ^ copy;
    		printf("PLAIN : %d | HASH : %s\n", rainbow[th], to_string(copy));
	}
	__syncthreads();
	else {
			
	}
}


int main(int argc, char** argv) {

	system("clear");
	
	unsigned int maxValue = 9999;
	unsigned int minValue = 1111;
	int rowCount = 64;
	int columnCount = 2;
	unsigned int *d_rainbow = NULL;
 	unsigned int *rainbow = NULL;

	int dev = findCudaDevice(argc, (const char **)argv);
	
	rainbow = (unsigned int *)malloc(sizeof(unsigned int) * rowCount * columnCount);

	printf("Generating data...\n");
    	srand(time(NULL));
    	for (int i = 0; i < rowCount; i+=columnCount)
    	{
		printf("%d\n", i);
        	rainbow[i] = rand() % (maxValue-minValue + 1) + minValue;
	}

	customCudaError(cudaMalloc((void **)&d_rainbow, sizeof(unsigned int) * rowCount * columnCount));
	customCudaError(cudaMemcpy(d_rainbow, rainbow, sizeof(unsigned int) * rowCount * columnCount, cudaMemcpyHostToDevice));
	
	hashFunction<<<1,rowCount>>>(d_rainbow, columnCount);
	customCudaError(cudaFree(d_rainbow));

	customCudaError(cudaDeviceSynchronize());

	free(rainbow);
    	exit(EXIT_SUCCESS);

}
