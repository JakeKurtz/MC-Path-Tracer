#pragma once

#include "GLCommon.h"
#include "RenderObject.h"
#include "Shader.h"
#include "Texture.h"
#include "Material.h"

#include <vector>
#include <map>
#include <bitset>

#include <cuda_runtime.h>
#include "dMatrix.cuh"
#include "dMath.cuh"
#include "Transform.h"
#include "Triangle.h"

#include <thrust/device_vector.h>
#include "Vertex.h"

/*
struct dTriangle
{
    __device__ bool intersect(const dRay& ray, float& u, float& v, float& t) const;
    __device__ bool hit(const dRay& ray, float& tmin, Isect& isect) const;
    __device__ bool hit(const dRay& ray) const;
    __device__ bool shadow_hit(const dRay& ray, float& tmin) const;

    float inv_area;
    dMaterial* material;
    dTransform* transform;
    dVertex v0, v1, v2;
    float3 face_normal;

    __host__ __device__ operator Triangle () const;
    __host__ __device__ dTriangle& operator= (const Triangle& v);
};
*/
/*
struct Triangle
{
    Triangle(Vertex v0, Vertex v1, Vertex v2) :
        v0(v0), v1(v1), v2(v2) {};

    float inv_area;
    Vertex v0, v1, v2;
    glm::vec3 face_normal;

    __host__ __device__ operator dTriangle () const;
    __host__ __device__ Triangle& operator= (const dTriangle& v);
};
*/
class Mesh : public RenderObject 
{
public:
    Mesh(
        const std::vector<std::shared_ptr<Triangle>> triangles,
        const std::shared_ptr<Material> material = std::make_shared<Material>(),
        const std::shared_ptr<Transform> transform = std::make_shared<Transform>());
    Mesh(
        const std::vector<Vertex> vertices, 
        const std::vector<uint32_t> indices,
        const std::shared_ptr<Material> material = std::make_shared<Material>(),
        const std::shared_ptr<Transform> transform = std::make_shared<Transform>());

    ~Mesh();

    glm::vec3 center_of_mass();

    void draw(Shader& shader);

    std::vector<Vertex> get_vertices();
    std::vector<unsigned int> get_indices();
    shared_ptr<Material> get_material();

    std::vector<std::shared_ptr<Triangle>> get_triangles();
    thrust::device_vector<dTriangle> get_device_triangles();

    int get_nmb_of_triangles();

    void set_triangles(std::vector<std::shared_ptr<Triangle>> triangles);

protected:
    // mesh data
    std::map<std::size_t, unsigned int> unique_vertices; // map<position-hash, index>

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    std::vector<std::shared_ptr<Triangle>> triangles;

    // device mesh data
    thrust::device_vector<uint32_t> d_indices;
    thrust::device_vector<dVertex> d_vertices;
    thrust::device_vector<dTriangle> d_triangles;

    // render data 
    unsigned int VAO;
    unsigned int VBO, EBO;
    int nmb_triangles = 0;

    void create_buffers();
    void delete_buffers();

    void create_dptr();
    void delete_dptr();

    void load_vertex(Vertex v);

private:
    int mesh_No;
    static int mesh_Cnt;
};

const size_t P1 = 948858202523369;
const size_t P2 = 984963996593221;
const size_t P3 = 220560183384551;

const size_t INV_CELL_SIZE = 1.f / 0.00000000000001;

float signed_volume_of_tetrahedron(glm::vec3 a, glm::vec3 b, glm::vec3 c);

glm::vec3 center_of_mass(shared_ptr<Mesh> mesh);
glm::vec3 center_of_mass(vector<Vertex> vertices, std::vector<unsigned int> indices);
glm::vec3 center_of_mass(vector<glm::vec3> vertices);

glm::vec3 center_of_geometry(vector<Vertex> vertices, std::vector<unsigned int> indices);
glm::vec3 center_of_geometry(vector<glm::vec3> vertices);