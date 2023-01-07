#include "Mesh.h"
#include "Isect.cuh"
#include "globals.h"
#include "CudaHelpers.h"
#include "mesh_initialization_kernels.cuh"

std::size_t teschner_hash(glm::vec3 v)
{
    size_t x = floorl((long double)v.x * INV_CELL_SIZE);
    size_t y = floorl((long double)v.y * INV_CELL_SIZE);
    size_t z = floorl((long double)v.z * INV_CELL_SIZE);

    return size_t((x * P1) ^ (y * P2) ^ (z * P3));
};

Mesh::Mesh(
    const std::vector<std::shared_ptr<Triangle>> triangles, 
    const std::shared_ptr<Material> material, 
    const std::shared_ptr<Transform> transform) 
    : triangles(triangles)
{
    (this)->material = material;
    (this)->transform = transform;
    (this)->name = gen_object_name("Mesh");

    for (auto &t : triangles) 
    {
        load_vertex(t->v0);
        load_vertex(t->v1);
        load_vertex(t->v2);
    }
    create_buffers();
    create_dptr();
}
Mesh::Mesh(
    const std::vector<Vertex> vertices, 
    const std::vector<uint32_t> indices,
    const std::shared_ptr<Material> material,
    const std::shared_ptr<Transform> transform) 
    : vertices(vertices), indices(indices)
{
    (this)->material = material;
    (this)->transform = transform;
    (this)->name = gen_object_name("Mesh");

    for (int i = 0; i < indices.size(); i+=3) 
    {
        auto i0 = indices[i];
        auto i1 = indices[i+1];
        auto i2 = indices[i+2];

        auto v0 = vertices[i0];
        auto v1 = vertices[i1];
        auto v2 = vertices[i2];

        triangles.push_back(std::make_shared<Triangle>(v0, v1, v2));
    }
    create_buffers();
    create_dptr();
}
Mesh::~Mesh()
{
    delete_buffers();
    delete_dptr();
}

glm::vec3 Mesh::center_of_mass()
{
    float volume = 0.f;
    glm::vec3 centroid = glm::vec3(0.f);

    for (int i = 0; i < indices.size(); i += 3) {
        auto iv0 = indices[i];
        auto iv1 = indices[i + 1];
        auto iv2 = indices[i + 2];

        auto v0 = vertices[iv0];
        auto v1 = vertices[iv1];
        auto v2 = vertices[iv2];

        float signed_volume = signed_volume_of_tetrahedron(v0.position, v1.position, v2.position);

        volume += signed_volume;
        centroid += signed_volume * (v0.position + v1.position + v2.position) / 4.f;
    }

    return centroid / volume;
}
void Mesh::draw(Shader& shader)
{
    material->send_uniforms(shader);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE0);
}

std::vector<Vertex> Mesh::get_vertices()
{
    return vertices;
}
std::vector<unsigned int> Mesh::get_indices()
{
    return indices;
}
shared_ptr<Material> Mesh::get_material()
{
    return material;
}
std::vector<std::shared_ptr<Triangle>> Mesh::get_triangles()
{
    return triangles;
}
thrust::device_vector<dTriangle> Mesh::get_device_triangles()
{
    return d_triangles;
}
int Mesh::get_nmb_of_triangles()
{
    return triangles.size();
}

void Mesh::set_triangles(std::vector<std::shared_ptr<Triangle>> triangles)
{
    (this)->triangles = triangles;

    vertices.clear();
    indices.clear();
    unique_vertices.clear();

    for (const auto& t : triangles) {
        load_vertex(t->v0);
        load_vertex(t->v1);
        load_vertex(t->v2);
    }
    create_buffers();
}

void Mesh::create_buffers()
{
    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    // vertex tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
    // vertex bitangent
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));

    glBindVertexArray(0);
}
void Mesh::delete_buffers()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void Mesh::create_dptr()
{
    d_triangles.resize(triangles.size());

    d_vertices.resize(vertices.size());
    thrust::copy(vertices.begin(), vertices.end(), d_vertices.begin());

    d_indices.resize(indices.size());
    thrust::copy(indices.begin(), indices.end(), d_indices.begin());

    init_mesh_on_device(
        triangles.size(), 
        thrust::raw_pointer_cast(d_vertices.data()), 
        thrust::raw_pointer_cast(d_indices.data()), 
        transform->get_dptr(), 
        material->get_dptr(), 
        thrust::raw_pointer_cast(d_triangles.data()));
}
void Mesh::delete_dptr()
{
}

void Mesh::load_vertex(Vertex v)
{
    auto hash = teschner_hash(v.position);

    if (unique_vertices.find(hash) == unique_vertices.end()) {
        // not found
        // push back the new index of vertex
        vertices.push_back(v);
        unsigned int index = vertices.size() - 1;

        unique_vertices.insert({ hash, index });
        indices.push_back(index);
    }
    else {
        // found
        // push back the index of existing vertex
        unsigned int index = unique_vertices.find(hash)->second;
        indices.push_back(index);
    }
}

float signed_volume_of_tetrahedron(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
    return glm::dot(a, glm::cross(b, c)) / 6.f;
}
glm::vec3 center_of_mass(shared_ptr<Mesh> mesh)
{
    float volume = 0.f;
    glm::vec3 centroid = glm::vec3(0.f);

    //for (auto mesh : model->get_meshes()) {

    auto indices = mesh->get_indices();
    auto vertices = mesh->get_vertices();

    for (int i = 0; i < indices.size(); i += 3) {
        auto iv0 = indices[i];
        auto iv1 = indices[i + 1];
        auto iv2 = indices[i + 2];

        auto v0 = vertices[iv0];
        auto v1 = vertices[iv1];
        auto v2 = vertices[iv2];

        float signed_volume = signed_volume_of_tetrahedron(v0.position, v1.position, v2.position);

        volume += signed_volume;
        centroid += signed_volume * (v0.position + v1.position + v2.position) / 4.f;
    }
    //}
    return centroid / volume;
}
glm::vec3 center_of_mass(vector<Vertex> vertices, std::vector<unsigned int> indices)
{

    float volume = 0.f;
    glm::vec3 centroid = glm::vec3(0.f);

    for (int i = 0; i < indices.size(); i += 3) {
        auto iv0 = indices[i];
        auto iv1 = indices[i + 1];
        auto iv2 = indices[i + 2];

        auto v0 = vertices[iv0];
        auto v1 = vertices[iv1];
        auto v2 = vertices[iv2];

        float signed_volume = signed_volume_of_tetrahedron(v0.position, v1.position, v2.position);

        volume += signed_volume;
        centroid += volume * (v0.position + v1.position + v2.position) / 4.f;
    }

    return centroid / volume;
}
glm::vec3 center_of_mass(vector<glm::vec3> vertices)
{
    return glm::vec3();
}
glm::vec3 center_of_geometry(vector<Vertex> vertices, std::vector<unsigned int> indices)
{
    return glm::vec3();
}
glm::vec3 center_of_geometry(vector<glm::vec3> vertices)
{
    return glm::vec3();
}
