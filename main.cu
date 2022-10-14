#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdlib.h>

const int N = 1<<24;
const int threadsPerBlock = 512;
const int blocksPerGrid = N/threadsPerBlock;

//kernel 1
__global__ void partial_dot(float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

//kernel 2
__global__ void atomic_dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    cache[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < threadsPerBlock; i++)
        {
            sum += cache[i];
        }
        atomicAdd(c, sum);
    }
}


int main (void) {

  cudaEvent_t start_k1, stop_k1, start_k2, stop_k2;
  cudaEventCreate(&start_k1); cudaEventCreate(&start_k2);
  cudaEventCreate(&stop_k1);  cudaEventCreate(&stop_k2);
  float milliseconds_k1 = 0, milliseconds_k2 = 0;

	float *a, *b, c, *t, *partial_c;
	float *dev_a, *dev_b, *dev_t, *dev_partial_c;
	
	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
    t = (float *)malloc(sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
	
	// allocate the memory on the gpu
	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
    cudaMalloc((void **)&dev_t, sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));
	
    float sumTest = 0;
	// fill in the host mempory with data
	for(int i=0; i<N; i++) {
	    a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        sumTest += a[i] * b[i];
	}
	printf("sumTest: %f\n",sumTest);
	
    *t = 0;

	// copy the arrays 'a' and 'b' to the gpu
	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t, t, sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_k1);
	partial_dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	
	// copy the array 'c' back from the gpu to the cpu
	cudaMemcpy(partial_c,dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
	
	// finish up on the cpu side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
		//printf("parial_c: %.6f \n", partial_c[i]);
	}
	cudaDeviceSynchronize();
    cudaEventRecord(stop_k1);
    cudaEventElapsedTime(&milliseconds_k1, start_k1, stop_k1);

	//#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Kernel 1 value %f ", c);
    printf("Time elapsed: %f \n", milliseconds_k1);
	
    cudaEventRecord(start_k2);
    atomic_dot<<< blocksPerGrid, threadsPerBlock >>>(dev_a, dev_b, dev_t);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_k2);
    cudaEventElapsedTime(&milliseconds_k2, start_k2, stop_k2);

    cudaMemcpy(t, dev_t, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Kernel 2: value is %f, total time elapsed %f\n", *t, milliseconds_k2);

	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	cudaFree(dev_t);
	
	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
	free(t);
}
