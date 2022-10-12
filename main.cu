#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512

__global__ void dot(int *a, int *b, int *c)
{
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

int main()
{
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(int);

   //allocate space for the variables on the device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, sizeof(int));

   //allocate space for the variables on the host
   a = (int *)malloc(size);
   b = (int *)malloc(size);
   c = (int *)malloc(sizeof(int));

   //this is our ground truth
   int sumTest = 0;
   //generate numbers
   for (int i = 0; i < N; i++)
   {
       a[i] = rand() % 10;
       b[i] = rand() % 10;
       sumTest += a[i] * b[i];
       printf(" %d %d \n",a[i],b[i]);
   }

   *c = 0;

   cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);

   dot<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(dev_a, dev_b,    dev_c);

   cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

   printf("%d ", *c);
   printf("%d ", sumTest);

   free(a);
   free(b);
   free(c);

   cudaFree(a);
   cudaFree(b);
   cudaFree(c);

   system("pause");

   return 0;

 }