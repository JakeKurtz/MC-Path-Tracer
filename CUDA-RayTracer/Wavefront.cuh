#pragma once
#include <vector_types.h>
#include <cstdint>
#include <cfloat>
#include "Isect.cuh"
#include "dRay.cuh"

struct Paths
{
    jek::Vec3f* f_light;
    jek::Vec3f* f_brdf;
    jek::Vec3f* f_sample;
    jek::Vec3f* Li_light;
    jek::Vec3f* Li_brdf;
    jek::Vec3f* beta;
    jek::Vec2f* pdf_light;
    jek::Vec2f* pdf_brdf;
    float* pdf_sample;
    dRay* ray;
    dRay* ray_light;
    uint32_t* len;
    uint32_t* light_id;
    Isect* isect;
    bool* dead;
    bool* visible;
};

struct Queues
{
    uint32_t*   new_path;
    uint32_t*   ext_path;
    uint32_t*   shadow_path;
    uint32_t*   mat_mix_path;

    uint32_t	new_path_len;
    uint32_t	ext_path_len;
    uint32_t	shadow_len;
    uint32_t	mat_mix_len;
};