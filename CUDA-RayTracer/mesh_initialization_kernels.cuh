#include "Triangle.h"
#include "BVH.h"

void init_mesh_on_device(
    const uint32_t size,
    const dVertex* vertices,
    const uint32_t* indices,
    dTransform* transform,
    dMaterial* material,
    dTriangle* triangles_out);

void init_BVH_triangle_info(
    const uint32_t size,
    const dTriangle* triangles,
    BVHPrimitiveInfo* BVH_triangle_info
);