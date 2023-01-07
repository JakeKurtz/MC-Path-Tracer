#pragma once
#include <cuda_runtime.h>

class dBRDF
{
public:
	__device__ virtual float3 sample_f(const dLight* light, const Isect isect, const float3 wo, float3& wi, float& cos_theta, float& pdf) = 0;
private:
	__device__ virtual float3 sample_wi(const Isect isect, const float3 wi, const float3 wo) = 0;
	__device__ virtual float pdf(const Isect isect, const float3 wo) = 0;
};