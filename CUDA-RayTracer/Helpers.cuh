#pragma once
#include <glm/glm.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <thrust/device_vector.h>

template<typename T> __device__ int binary_search(T* list, int size, const float& val);
template<typename T> __device__ int upper_bound(T* list, int size, const float& val);
template<typename T> __device__ int lower_bound(T* list, int size, const float& val);

__device__ int binary_search(float* list, int size, const float& val);
__device__ int upper_bound(float* list, int size, const float val);
__device__ int lower_bound(float* list, int size, const float val);

__device__ uint32_t get_thread_id();
__device__ uint32_t atomicAggInc(uint32_t* ctr);