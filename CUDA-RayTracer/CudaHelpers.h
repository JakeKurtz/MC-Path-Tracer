#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "dMath.cuh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

extern void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);
