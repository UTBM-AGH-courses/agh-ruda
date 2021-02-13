#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <numeric>
#include <string>

using namespace std;

#define MAX_BLOCK_SIZE 1024
#define MAX_ITERATION 100000

__device__ bool found;
__device__ unsigned int iteration;

cudaError_t customCudaError(cudaError_t result)
{
	if (result != cudaSuccess)
    	{
        	fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	        assert(result == cudaSuccess);
    	}
   	return result;
}

void displayTable(unsigned int *plain, unsigned int *hash, unsigned int columnCount, unsigned int rowCount)
{
	printf("Rainbow table (row=%d x depth=%d) :\n", rowCount, columnCount);
	for(int i = 0; i < rowCount; i++)
	{
		printf("PLAIN : %d | HASH : %d\n", plain[i], hash[i]);
	}
}

__device__
void hashingKernel(unsigned int plain, unsigned int *hash)
{
	// Hashing kernel (36669 => 174576660)
	*hash = ((plain >> 16) ^ plain) * 0x45;
	*hash = ((*hash >> 16) ^ *hash) * 0x45;
	*hash = (*hash >> 16) ^ *hash;
}

__device__
void reductionKernel(unsigned int maxValue, unsigned int hash, unsigned int *reduction)
{
	// Reduction kernel (174576660 => 17457)
	while (hash > maxValue)
	{
		hash = hash / 10;
	}
	*reduction = hash;
}

__global__
void findingKernel(unsigned int *plainArray, unsigned int *hashArray, unsigned int hash, unsigned int columnCount, unsigned int rowCount, unsigned int maxValue)
{
	int th = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int localHash = hashArray[th];
	unsigned int plain = plainArray[th];
	unsigned int reduction;

	while (!found)
	{
		// If the hash we are trying to guess match one of the entry from the hashArray
		if (localHash == hash)
		{
			found = true;
			__threadfence();
			for (int i = 0; i < columnCount; i++)
			{
				// Hash the plain text of the corresponding row
				hashingKernel(plain, &localHash);
				if (localHash == hash)
				{
					printf("#### Match for %d (HASH : %d) on Thread %d ####\n", plain, localHash, th);
					__syncthreads();
					break;
				}
				// If both hashes do not match, reduce and hash until getting the correct value
				else
				{
					reduction = localHash;
					reductionKernel(maxValue, localHash, &reduction);
					plain = reduction;
					__syncthreads();
				}
			}

		}
		// If no entries match, reduce and hash until getting a matching value
		else
		{
			iteration++;
			__threadfence();
			if (iteration >= MAX_ITERATION)
			{
				if (th == 0) printf("#### Sorry, no match for this hash :( ####\n");
				break;
			}
			reductionKernel(maxValue, localHash, &reduction);
			plain = reduction;
			hashingKernel(plain, &localHash);
			reduction = localHash;
		}
	}

}

__global__
void rainbowKernel(unsigned int *plainArray, unsigned int *hashArray, unsigned int columnCount, unsigned int maxValue) {
	int th = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int plain = plainArray[th];
	unsigned int hash;
	unsigned int reduction;

	for (int i = 0; i < columnCount; i++)
	{
		hashingKernel(plain, &hash);
		reduction = hash;
		reductionKernel(maxValue, hash, &reduction);
		plain = reduction;
	}
	hashArray[th] = hash;
}

void rainbowWrapper(unsigned int rowCount, unsigned int columnCount, unsigned int maxValue, unsigned int *plainArray, unsigned int *hashArray, bool display, unsigned int hash)
{
	unsigned int *d_plainArray = NULL;
	unsigned int *d_hashArray = NULL;
	cudaEvent_t start;
	cudaEvent_t stop;

	// Allocate memory space on the device
	customCudaError(cudaMalloc((void **)&d_plainArray, sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE));
	customCudaError(cudaMalloc((void **)&d_hashArray, sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE));

	// Copy data on the device
	customCudaError(cudaMemcpy(d_plainArray, plainArray, sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE, cudaMemcpyHostToDevice));

	// Lauch the rainbow table generation kernel
	rainbowKernel<<<rowCount,1024>>>(d_plainArray, d_hashArray, columnCount, maxValue);
	customCudaError(cudaDeviceSynchronize());

	// Fetch the data from the device
	customCudaError(cudaMemcpy(hashArray, d_hashArray, sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE, cudaMemcpyDeviceToHost));

	// Record the start event for the second kernel
	customCudaError(cudaEventCreate(&start));
	customCudaError(cudaEventCreate(&stop));
	customCudaError(cudaEventRecord(start, NULL));

	// Launch the hash resolver kernel
	printf("Searching for the hash into the table...\n");
	findingKernel<<<rowCount,1024>>>(d_plainArray, d_hashArray, hash, columnCount, rowCount, maxValue);
	customCudaError(cudaDeviceSynchronize());

	// Display the table (or not)
	if (display == 1)
	{
		displayTable(plainArray, hashArray, columnCount, rowCount);
	}

	// Record the stop event for the first event
	customCudaError(cudaEventRecord(stop, NULL));
	customCudaError(cudaEventSynchronize(stop));

	// Display the time enlapsed informations
	printf("################\n");
	float msecTotal = 0.0f;
	customCudaError(cudaEventElapsedTime(&msecTotal, start, stop));
	double gigaFlops = (columnCount * rowCount * MAX_BLOCK_SIZE * 1.0e-9f) / (msecTotal / 1000.0f);
	printf("Cuda processing time = %.3fms, Performance = %.3f GFlop/s\n", msecTotal, gigaFlops);

	customCudaError(cudaFree(d_plainArray));
	customCudaError(cudaFree(d_hashArray));
}

int main(int argc, char** argv) {

	unsigned int maxValue = 99999;
	unsigned int minValue = 11111;
	unsigned int rowCount = 4;
	unsigned int columnCount = 4096;
	bool display = false;
	char * s_hash;
	unsigned int hash = 0;
 	unsigned int *plainArray = NULL;
	unsigned int *hashArray = NULL;

	// Get the device
	int dev = findCudaDevice(argc, (const char **)argv);

	// Clear the terminal
	system("clear");

	// Display the help
	if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
       	printf("Usage :\n");
		printf("      -hash=HASH [0] (Password hash you want to crack) \n");
		printf("      -block=BLOCK [4] (Rainbow table's row count (1 block = 1024 row)) \n");
		printf("      -depth=DEPTH [4096] (Rainbow table's column count) \n");
       	printf("      -verbose (Display the rainbow table)\n");

       	exit(EXIT_SUCCESS);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "hash"))
    {
       	getCmdLineArgumentString(argc, (const char **)argv, "hash", &s_hash);
       	hash = atoi(s_hash);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "block"))
    {
		rowCount = getCmdLineArgumentInt(argc, (const char**)argv, "block");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "depth"))
	{
		columnCount = getCmdLineArgumentInt(argc, (const char**)argv, "depth");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
	{
		display = true;
	}

	plainArray = (unsigned int *)malloc(sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE);
	hashArray = (unsigned int *)malloc(sizeof(unsigned int) * rowCount * MAX_BLOCK_SIZE);

	printf("######## RUDA - Rainbow table password cracking CUDA implemetation ########\n");
	printf("Choosen parameters :\n");
	printf("      HASH = %d\n", hash);
	printf("      ROWS = %d\n", rowCount * MAX_BLOCK_SIZE);
	printf("      COLUMNS = %d\n", columnCount);
	printf("      MAX_ITERATION = %d\n", MAX_ITERATION);


	printf("\nGenerating random passwords...\n");
    srand(time(NULL));
    for (int i = 0; i < rowCount * MAX_BLOCK_SIZE; i++)
  	{
		plainArray[i] = rand() % (maxValue-minValue + 1) + minValue;
	}

	printf("Generation done\n");

	rainbowWrapper(rowCount, columnCount, maxValue, plainArray, hashArray, display, hash);

	free(plainArray);
	free(hashArray);
    exit(EXIT_SUCCESS);

}
