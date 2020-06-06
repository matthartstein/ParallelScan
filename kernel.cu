/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *inout, unsigned size, float *sum)
{
    // Perform a local scan on 2*BLOCK_SIZE items
    
    __shared__ float temp[BLOCK_SIZE * 2];
    
    int location = 0;
    int thread = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x;
    
    if(index + thread < size)
        temp[thread] = inout[index + thread];
    else
        temp[thread] = 0;
        
    if(index + thread + blockDim.x < size)
        temp[thread + blockDim.x] = inout[index + thread + blockDim.x];
    else
        temp[thread + blockDim.x] = 0;
        
    __syncthreads();
    
    int offset = 1;
    while(offset <= blockDim.x) {
        location = (thread + 1) * 2 * offset - 1;
        if(location < (2 * BLOCK_SIZE))
            temp[location] += temp[location - offset];
        
        offset *= 2;
        __syncthreads();
    }
    
    if(thread == 0) {
        if(sum != NULL)
            sum[blockIdx.x] = temp[2 * blockDim.x - 1];
        
        temp[2 * blockDim.x - 1] = 0;
    }
    
    __syncthreads();
    
    location = 0;
    float val = 0;
    offset = blockDim.x;
    
    while(offset > 0) {
        location = (2 * offset * (thread + 1)) - 1;
        if(location < 2 * BLOCK_SIZE) {
            val = temp[location];
            temp[location] += temp[location - offset];
            temp[location - offset] = val;
        }
        offset >>= 1;
        __syncthreads();
    }
    
    if(index + thread < size)
        inout[index + thread] = temp[thread];
    
    if(index + thread + blockDim.x < size)
        inout[index + thread + blockDim.x] = temp[thread + blockDim.x];
}

__global__ void addKernel(float *inout, float *sum, unsigned size)
{
    // Use the scan of partial sums to update 2*BLOCK_SIZE items
    
    int block = blockIdx.x;
    int thread = threadIdx.x;
    int location = 2 * blockDim.x * block + thread;
    
    if(location < 2 * BLOCK_SIZE)
        inout[location] += sum[blockIdx.x];
        
    if(location + blockDim.x < 2 * BLOCK_SIZE)
        inout[location + blockDim.x] += sum[blockIdx.x];
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *inout, unsigned in_size)
{
	float *sum;
	unsigned num_blocks;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;

	num_blocks = in_size/(BLOCK_SIZE*2);
	if(in_size%(BLOCK_SIZE*2) !=0) num_blocks++;

	dim_block.x = BLOCK_SIZE; dim_block.y = 1; dim_block.z = 1;
	dim_grid.x = num_blocks; dim_grid.y = 1; dim_grid.z = 1;

	if(num_blocks > 1) {
		cuda_ret = cudaMalloc((void**)&sum, num_blocks*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

		preScanKernel<<<dim_grid, dim_block>>>(inout, in_size, sum);
		preScan(sum, num_blocks);
		addKernel<<<dim_grid, dim_block>>>(inout, sum, in_size);

		cudaFree(sum);
	}
	else
		preScanKernel<<<dim_grid, dim_block>>>(inout, in_size, NULL);
}