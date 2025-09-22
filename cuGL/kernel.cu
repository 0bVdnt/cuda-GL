 #include "kernel.h"
#include <stdio.h>

// CUDA kernel to compute colors
__global__ void colorKernel(float4* pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{ 
		unsigned int i = y * width + x;

		float r = 0.5f + 0.5f * sinf(x * 0.1f + time);
		float g = 0.5f + 0.5f * sinf(y * 0.1f + time);
		float b = 0.5f + 0.5f * cosf(x * 0.1f + y * 0.1f + time);

		// Using the Z component of float4 to store the red color value for simplicity
		pos[i].z = r;
	}
}

// Wrapper function called from C++
void launch_kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	colorKernel << <numBlocks, threadsPerBlock >> > (pos, width, height, time);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
	}
}