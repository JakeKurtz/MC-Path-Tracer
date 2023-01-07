#pragma once
#include <cuda_runtime.h>
#include "dMaterial.cuh"
struct _ALIGN(128) Isect
{
    jek::Vec3f	    position    = jek::Vec3f(0.f);
    jek::Vec3f	    normal      = jek::Vec3f(0.f);
    jek::Vec3f	    tangent     = jek::Vec3f(0.f);
    jek::Vec3f	    bitangent   = jek::Vec3f(0.f);
    jek::Vec2f	    texcoord    = jek::Vec2f(0.f);
    float		    t           = 0.f;
    bool		    was_found   = false;
    uint32_t        tri_id;
    //uint32_t        mat_id;
    dMaterial*       material;
};