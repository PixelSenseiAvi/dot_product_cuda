# dot_product_cuda
CUDA Implementation of dot product

The dot product of two vectors a = and b written a • b, is simply the sum of the component-by-component products:

a •b = summation[ai x bi]


Problem Statement:
1. Write a CUDA code to compute in parallel the dot product of two random single precision floating-point vectors with size N = 1<<24;
2. Write two kernel functions for the dot product computation on GPU:
• 1) kernel1: use shared memory and parallel reduction to calculate partial sum on each thread block. (Add up all the partial sums on CPU after transferring all the partial sums back to host from device)
• 2) kernel2: use shared memory, parallel reduction, and atomic function or atomic lock to perform the entire computation on GPU. (Transfer the final dot product result back to host from device)
3. Compare the time it takes for kernell and kernel 2. (Use cudaEventRecord() for the timing.)

