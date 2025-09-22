#pragma once

#include <cuda_runtime.h>
#include <vector_types.h> // For float4

void launch_kernel(float4* pos, unsigned int width, unsigned int height, float time);