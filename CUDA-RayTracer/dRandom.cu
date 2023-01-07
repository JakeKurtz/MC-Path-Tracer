#include "dRandom.cuh"
#include "dMath.cuh"

__constant__ unsigned int shift1[4] = { 6, 2, 13, 3 };
__constant__ unsigned int shift2[4] = { 13, 27, 21, 12 };
__constant__ unsigned int shift3[4] = { 18, 2, 7, 13 };
__constant__ unsigned int offset[4] = { 4294967294, 4294967288, 4294967280, 4294967168 };
__shared__ unsigned int randStates[32];

__device__ uint32_t wang_hash(uint32_t seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

__device__ uint32_t lowerbias32(uint32_t x)
{
	x ^= x >> 16;
	x *= 0xa812d533;
	x ^= x >> 15;
	x *= 0xb278e4ad;
	x ^= x >> 17;
	return x;
}

__device__ float random()
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int seed = ((i * 1024 + j) * 100) + clock();
	return lowerbias32(seed) * (1.0 / 4294967296.0);
};

__device__ float rand_float()
{
	return random();
};

__device__ float rand_float(float min, float max)
{
	return ((random() * (max - min)) + min);
};

__device__ int rand_int(int min, int max)
{
	return min + (int)(random() * INT_MAX) / (INT_MAX / (max - min + 1) + 1);
};

__device__ int rand_int()
{
	return (int)(random() * INT_MAX);
};

__device__ float2 UniformSampleSquare() {
	return make_float2(random(), random());
}

__device__ float3 UniformSampleHemisphere(const float2& u) {
	float z = u.x;
	float r = sqrt(fmaxf((float)0, (float)1. - z * z));
	float phi = 2 * M_PI * u.y;
	return make_float3(r * cos(phi), r * sin(phi), z);
}

__device__ float3 UniformSampleSphere(const float3& u) {
	float z = 1 - 2 * u.x;
	float r = sqrt(fmaxf((float)0, (float)1 - z * z));
	float phi = 2 * M_PI * u.y;
	return make_float3(r * cos(phi), r * sin(phi), z);
}

__device__ float3 UniformSampleSphere()
{
	float2 u = make_float2(random(), random());

	float z = 1 - 2 * u.x;
	float r = sqrt(fmaxf((float)0, (float)1 - z * z));
	float phi = 2 * M_PI * u.y;
	return make_float3(r * cos(phi), r * sin(phi), z);
}

__device__ float2 UniformSampleDisk(const float2& u) {
	float r = sqrt(u.x);
	float theta = 2 * M_PI * u.y;
	return make_float2(r * cos(theta), r * sin(theta));
}

__device__ float2 ConcentricSampleDisk(const float2& u) {
	// Map uniform random numbers to
	float2 uOffset = 2.f * u - make_float2(1, 1);

	// Handle degeneracy at the origin 
	if (uOffset.x == 0 && uOffset.y == 0)
		return make_float2(0, 0);

	// Apply concentric mapping to point >>
	float theta, r;
	if (abs(uOffset.x) > abs(uOffset.y)) {
		r = uOffset.x;
		theta = M_PI_4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
	}
	return r * make_float2(cos(theta), sin(theta));
}

__device__ float2 ConcentricSampleDisk()
{
	float2 u = make_float2(random(), random());

	// Map uniform random numbers [0, 1] to [-1, 1]
	float2 uOffset = 2.f * u - make_float2(1, 1);

	// Handle degeneracy at the origin 
	if (uOffset.x == 0 && uOffset.y == 0)
		return make_float2(0, 0);

	// Apply concentric mapping to point >>
	float theta, r;
	if (abs(uOffset.x) > abs(uOffset.y)) {
		r = uOffset.x;
		theta = M_PI_4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
	}
	return r * make_float2(cos(theta), sin(theta));
}

__device__ inline float3 CosineSampleHemisphere(const float2& u)
{
	float2 d = ConcentricSampleDisk(u);
	float z = sqrt(fmaxf((float)0, 1 - d.x * d.x - d.y * d.y));
	return make_float3(d.x, d.y, z);
}

__device__ inline float3 CosineSampleHemisphere()
{
	float2 d = ConcentricSampleDisk();
	float z = sqrt(fmaxf((float)0, 1 - d.x * d.x - d.y * d.y));
	return make_float3(d.x, d.y, z);
}