#include "Helpers.cuh"

__device__ int binary_search(float* list, int size, const float& val)
{
	int middle, left = 0, right = size;
	while (right >= 1) {
		middle = (right - left) / 2 + left;

		if (val == middle) return middle;
		else if (val < middle) right = middle - 1;
		else left = middle + 1;
	}
	return -1;
}
__device__ int upper_bound(float* list, int size, const float val)
{
	int middle, left = 0, right = size;
	while (left < right) {
		middle = (right - left) / 2 + left;

		if (val >= list[middle]) {
			left = middle + 1;
		}
		else {
			right = middle;
		}
	}
	if (left < size && list[left] <= val) left++;
	return left;
}

__device__ int lower_bound(float* list, int size, const float val)
{
	int middle, left = 0, right = size;
	while (left < right) {
		middle = (right - left) / 2 + left;

		if (val <= list[middle]) {
			right = middle;
		}
		else {
			left = middle + 1;
		}
	}
	if (left < size && list[left] < val) left++;
	return left;
}

__device__ uint32_t atomicAggInc(uint32_t* ctr)
{
	//uint32_t mask = __ballot(1);
	uint32_t mask = __activemask();
	uint32_t leader = __ffs(mask) - 1;
	uint32_t laneid = threadIdx.x % 32;
	uint32_t res;

	if (laneid == leader)
		res = atomicAdd(ctr, __popc(mask));

	res = __shfl_sync(mask, res, leader);
	//res = __shf(res, leader);
	return res + __popc(mask & ((1 << laneid) - 1));
}

__device__ uint32_t get_thread_id() 
{ 
	return threadIdx.x + blockIdx.x * blockDim.x; 
}
/*
__device__ uint32_t get_thread_id()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
*/