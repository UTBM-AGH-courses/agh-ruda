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

void displayTable(unsigned long long *rainbow, unsigned int columnCount, unsigned int rowCount)
{
	printf("Here the generated table (r:%d x c:%d) :\n", rowCount, columnCount);
	for(int i = 0; i < rowCount; i++)
	{
		printf("PLAIN : %d | HASH : %llu\n", rainbow[i*columnCount], rainbow[i*columnCount + 1]);
	}
}

__global__
void findingKernel(unsigned long long *rainbow, unsigned long long hash, unsigned int columnCount, unsigned int rowCount)
{
	bool found = false;

	for(int i = 0; i < rowCount; i++)
	{
		if (rainbow[i*columnCount + 1] != hash)
		{
			printf("NOK\n");
		}
		else
		{
			printf("OKKKKKKKKKKKKKKKKKKKKKKK\n");
		}
	}

}

__global__
void rainbowKernel(unsigned long long *rainbow, unsigned int columnCount, unsigned int maxValue)
{
	int th = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long plain = rainbow[th*columnCount];
	unsigned long long hash;
	unsigned long long reduction;

	for (int i = 0; i < columnCount; i++)
	{
		// HASHING
		hash = ((plain >> 8) ^ plain) * 0x45d9f3b;
		hash = ((hash >> 8) ^ hash) * 0x45d9f3b;
		hash = (hash >> 8) ^ hash;

		reduction = hash;
		// REDUCTION
		while (reduction > maxValue)
		{
			reduction = reduction / 10;
		}
		plain = reduction;
	}
	rainbow[th*columnCount + 1] = hash;
}

int main(int argc, char** argv) {

	system("clear");

	unsigned int maxValue = 9999;
	unsigned int minValue = 1111;
	int rowCount = 32;
	int columnCount = 2;
	char *s_hash = NULL;
	unsigned long long hash = 0;
	unsigned long long *d_rainbow = NULL;
 	unsigned long long *rainbow = NULL;
	unsigned long long *f_rainbow = NULL;

	int dev = findCudaDevice(argc, (const char **)argv);

	rainbow = (unsigned long long *)malloc(sizeof(unsigned long long) * rowCount * columnCount);
	f_rainbow = (unsigned long long *)malloc(sizeof(unsigned long long) * rowCount * columnCount);

	if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?"))
    	{
        	printf("Usage :\n");
		printf("      -hash=HASH [2568782378648878273] (Password hash you want to crack) \n");
        	printf("      -verbose (Display the rainbow table)\n");

        	exit(EXIT_SUCCESS);
    	}

	if (checkCmdLineFlag(argc, (const char **)argv, "hash"))
    	{
        	getCmdLineArgumentString(argc, (const char **)argv, "hash", &s_hash);
        	hash = atoll(s_hash);
		printf("%llu\n", hash);
    	}

	printf("Generating data...\n");
    	srand(time(NULL));
    	for (int i = 0; i < rowCount; i++)
    	{
        	rainbow[i*columnCount] = rand() % (maxValue-minValue + 1) + minValue;
	}
	rainbow[62] = 1234;
	printf("Generation done.\n");

	customCudaError(cudaMalloc((void **)&d_rainbow, sizeof(unsigned long long) * rowCount * columnCount));
	customCudaError(cudaMemcpy(d_rainbow, rainbow, sizeof(unsigned long long) * rowCount * columnCount, cudaMemcpyHostToDevice));

	rainbowKernel<<<1,rowCount>>>(d_rainbow, columnCount, maxValue);

	customCudaError(cudaMemcpy(f_rainbow, d_rainbow, sizeof(unsigned long long) * rowCount * columnCount, cudaMemcpyDeviceToHost));
	displayTable(f_rainbow, columnCount, rowCount);

	findingKernel<<<1,1>>>(d_rainbow, hash, columnCount, rowCount);


	customCudaError(cudaFree(d_rainbow));
	customCudaError(cudaDeviceSynchronize());

	free(rainbow);
	free(f_rainbow);
    	exit(EXIT_SUCCESS);

}
