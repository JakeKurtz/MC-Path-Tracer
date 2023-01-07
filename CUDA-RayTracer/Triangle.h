#pragma once
#include "GLCommon.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "Vertex.h"
#include "Material.h"
#include "Transform.h"
#include "dRay.cuh"

struct dTriangle
{
    __device__ bool intersect(const dRay& ray, float& u, float& v, float& t) const;
    __device__ bool hit(const dRay& ray, float& tmin, Isect& isect) const;
    __device__ bool hit(const dRay& ray) const;
    __device__ bool shadow_hit(const dRay& ray, float& tmin) const;

    float inv_area;
    dMaterial* material = nullptr;
    dTransform* transform = nullptr;
    dVertex v0, v1, v2;
    jek::Vec3f face_normal;
};

struct Triangle
{
public:
    Triangle();
    Triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2);

    float inv_area;
    Vertex v0, v1, v2;
    glm::vec3 face_normal;

private:
    void init();
};

__device__ void intersect(const LinearBVHNode* __restrict__ nodes, const dTriangle* __restrict__ triangles, const dRay ray, Isect& isect);
__device__ bool intersect_shadows(const LinearBVHNode* __restrict__ nodes, const dTriangle* __restrict__ triangles, const dRay ray, float& tmin);