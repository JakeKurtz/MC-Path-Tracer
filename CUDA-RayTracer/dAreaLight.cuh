/*
#pragma once

#include "dLight.cuh"

class GeometricObj;
class Isect;
class dRay;
class LinearBVHNode;
class dTriangle;

class dAreaLight : public dLight
{
public:
	__device__ dAreaLight(void);
	__device__ virtual void get_direction(const Isect isect, float3& wi, float3& sample_point) const;
	__device__ virtual float3 L(const Isect isect, const float3 wi, const float3 sample_point) const;
	__device__ virtual float get_pdf(const Isect isect) const;
	__device__ virtual float get_pdf(const Isect isect, const float3 wi) const;
	__device__ virtual bool visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay ray) const;
	__device__ void set_position(const float x, const float y, const float z);
	__device__ void set_position(const float3 pos);
	__device__ void set_object(GeometricObj* obj_ptr);
	//__device__ void set_material(Emissive* mat_ptr);
private:
	float3			position;
	GeometricObj*	object_ptr;
};
*/