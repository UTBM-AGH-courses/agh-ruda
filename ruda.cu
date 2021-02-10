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
	printf("Rainbow table (row=%d x depth=%d) :\n", rowCount, columnCount);
	for(int i = 0; i < rowCount; i++)
	{
		printf("PLAIN : %d | HASH : %llu\n", rainbow[i*columnCount], rainbow[i*columnCount + 1]);
	}
}

__global__
void findingKernel(unsigned long long *rainbow, unsigned long long hash, unsigned int columnCount, unsigned int rowCount)
{
	bool found = false;
	printf("############\n");
	printf("Finding the password into the rainbow table : \n");

	for(int i = 0; i < rowCount; i++)
	{
		if (rainbow[i*columnCount + 1] == hash)
		{
			printf("Match for %llu (HASH : %llu)\n", rainbow[i*columnCount], hash);
		}
	}
	printf("############\n");
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
		hash = ((plain >> 16) ^ plain) * 0x45d;
		hash = ((hash >> 16) ^ hash) * 0x45d;
		hash = (hash >> 16) ^ hash;

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
	unsigned int rowCount = 32;
	unsigned int columnCount = 4096;
	char *s_hash = NULL;
	int display = 0;
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
		printf("      -hash=HASH [0] (Password hash you want to crack) \n");
		printf("      -row=ROW [32] (Rainbow table's row count) \n");
		printf("      -depth=DEPTH [4096] (Rainbow table's column count) \n");
        	printf("      -verbose (Display the rainbow table)\n");

        	exit(EXIT_SUCCESS);
    	}

	if (checkCmdLineFlag(argc, (const char **)argv, "hash"))
    	{
        	getCmdLineArgumentString(argc, (const char **)argv, "hash", &s_hash);
        	hash = atoll(s_hash);
    	}

        if (checkCmdLineFlag(argc, (const char **)argv, "row"))
        {
		rowCount = getCmdLineArgumentInt(argc, (const char**)argv, "row");

	}
        if (checkCmdLineFlag(argc, (const char **)argv, "depth"))
        {
		columnCount = getCmdLineArgumentInt(argc, (const char**)argv, "depth");
        }

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    	{
        	display = 1;
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
