#pragma once
#include <limits>
#include <cuda_runtime.h>

#include "dRay.cuh"

//__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

class Bounds3f {
public:
	__host__ __device__ Bounds3f() {
		float minNum = std::numeric_limits<float>::lowest();
		float maxNum = std::numeric_limits<float>::max();
		pMin = jek::Vec3f(maxNum, maxNum, maxNum);
		pMax = jek::Vec3f(minNum, minNum, minNum);
	}

	__host__ __device__ Bounds3f(const jek::Vec3f& p) : pMin(p), pMax(p) {};

	__host__ __device__ Bounds3f(const jek::Vec3f& p1, const jek::Vec3f& p2)
	{
		pMin = jek::Vec3f(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
		pMax = jek::Vec3f(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
	};

	__host__ __device__ const jek::Vec3f& __restrict__ operator[](int i) const
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ jek::Vec3f& __restrict__ operator[](int i)
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ bool operator==(const Bounds3f& b) const
	{
		return b.pMin == pMin && b.pMax == pMax;
	};

	__host__ __device__ bool operator!=(const Bounds3f& b) const
	{
		return b.pMin != pMin || b.pMax != pMax;
	};

	__host__ __device__ jek::Vec3f corner(int corner) const {
		return jek::Vec3f(
			(*this)[corner & 1].x,
			(*this)[(corner & 2) ? 1 : 0].y,
			(*this)[(corner & 4) ? 1 : 0].z
		);
	};

	__host__ __device__ jek::Vec3f diagonal() const { return pMax - pMin; };

	__host__ __device__ double surface_area() const
	{
		jek::Vec3f d = diagonal();
		return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	__host__ __device__ double volume() const {
		jek::Vec3f d = diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ int maximum_extent() const {
		jek::Vec3f d = diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ jek::Vec3f offset(const jek::Vec3f& p) const {
		jek::Vec3f o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
		if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
		return o;
	}

	__device__ bool hit(const dRay& __restrict__ ray) const
	{
		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = (pMin.x - ray.o.x) / ray.d.x;
		tmax = (pMax.x - ray.o.x) / ray.d.x;
		tymin = (pMin.y - ray.o.y) / ray.d.y;
		tymax = (pMax.y - ray.o.y) / ray.d.y;

		if ((tmin > tymax) || (tymin > tmax))
			return false;
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;

		tzmin = (pMin.z - ray.o.z) / ray.d.z;
		tzmax = (pMax.z - ray.o.z) / ray.d.z;

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		//float tminbox = 1.f;//min_max(tmin, tmax, min_max(tymin, tymax, min_max(tzmin, tzmax, 0)));
		//float tmaxbox = 1.f;//max_min(tmin, tmax, max_min(tymin, tymax, max_min(tzmin, tzmax, K_HUGE)));

		//return (tminbox <= tmaxbox);
		return true;
	};

	__device__ inline bool hit(const dRay& __restrict__ ray, const jek::Vec3f& __restrict__ invDir, const int dirIsNeg[3]) const
	{
		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = ((*this)[dirIsNeg[0]].x - ray.o.x) * invDir.x;
		tmax = ((*this)[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
		tymin = ((*this)[dirIsNeg[1]].y - ray.o.y) * invDir.y;
		tymax = ((*this)[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;

		if ((tmin > tymax) || (tymin > tmax))
			return false;
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;

		tzmin = ((*this)[dirIsNeg[2]].z - ray.o.z) * invDir.z;
		tzmax = ((*this)[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		//float tminbox = 1.f;//min_max(tmin, tmax, min_max(tymin, tymax, min_max(tzmin, tzmax, 0)));
		//float tmaxbox = 1.f;//max_min(tmin, tmax, max_min(tymin, tymax, max_min(tzmin, tzmax, K_HUGE)));

		//return (tminbox <= tmaxbox);

		return true;
	}

	jek::Vec3f pMin, pMax;
};