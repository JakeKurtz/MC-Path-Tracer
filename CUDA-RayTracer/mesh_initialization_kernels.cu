#include "light_initialization_kernels.cuh"
#include "Vertex.h"
#include "Transform.h"
#include "Triangle.h"

__global__
void g_init_mesh_on_device(
    const uint32_t size,
    const dVertex* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    dTransform* transform, 
    dMaterial* material, 
    dTriangle* triangles_out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) 
    {
        int j = i * 3;

        auto iv0 = indices[j];
        auto iv1 = indices[j + 1];
        auto iv2 = indices[j + 2];

        auto v0 = vertices[iv0];
        auto v1 = vertices[iv1];
        auto v2 = vertices[iv2];

        auto v0v1 = v1.position - v0.position;
        auto v0v2 = v2.position - v0.position;
        auto ortho = cross(v0v1, v0v2);

        auto area = length(ortho) * 0.5;
        auto inv_area = 1.0 / area;

        jek::Vec3f face_normal = normalize(ortho);

        triangles_out[i].v0 = v0;
        triangles_out[i].v1 = v1;
        triangles_out[i].v2 = v2;
        triangles_out[i].inv_area = inv_area;
        triangles_out[i].face_normal = face_normal;
        triangles_out[i].material = material;
        triangles_out[i].transform = transform;
    }
}
void init_mesh_on_device(
    const uint32_t size,
    const dVertex* vertices,
    const uint32_t* indices,
    dTransform* transform,
    dMaterial* material,
    dTriangle* triangles_out)
{
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    g_init_mesh_on_device <<< num_blocks, block_size >>> (size, vertices, indices, transform, material, triangles_out);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void g_init_BVH_triangle_info(
    const uint32_t size, 
    const dTriangle* __restrict__ triangles, 
    BVHPrimitiveInfo* BVH_triangle_info)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride)
    {
        auto t = triangles[i];
        auto matrix = t.transform->matrix;

        jek::Vec3f v0 = jek::Vec3f(matrix * jek::Vec4f(t.v0.position, 1.f));
        jek::Vec3f v1 = jek::Vec3f(matrix * jek::Vec4f(t.v1.position, 1.f));
        jek::Vec3f v2 = jek::Vec3f(matrix * jek::Vec4f(t.v2.position, 1.f));

        BVH_triangle_info[i] = BVHPrimitiveInfo(i, Union(Bounds3f(v0, v1), v2));
    }
}
void init_BVH_triangle_info(
    const uint32_t size, 
    const dTriangle* triangles, 
    BVHPrimitiveInfo* BVH_triangle_info)
{
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    g_init_BVH_triangle_info <<< num_blocks, block_size >>> (size, triangles, BVH_triangle_info);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
