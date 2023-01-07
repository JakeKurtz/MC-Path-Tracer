#include "Scene.h"
#include "massert.h"
#include "CudaHelpers.h"
#include "mesh_initialization_kernels.cuh"
#include <thrust/gather.h>
#define LOG

static inline glm::mat4 mat4_cast(const aiMatrix4x4& m) { return glm::transpose(glm::make_mat4(&m.a1)); }
static inline glm::mat4 mat4_cast(const aiMatrix3x3& m) { return glm::transpose(glm::make_mat3(&m.a1)); }

Scene::Scene()
{
    this->environment_light = std::make_shared<EnvironmentLight>(glm::vec3(0.8f));
}

Scene::Scene(vector<std::shared_ptr<RenderObject>> models, vector<std::shared_ptr<Light>> lights, std::shared_ptr<Camera> camera)
{
    //models = _models;
    //lights = _lights;
    //set_camera(camera);
    this->environment_light = std::make_shared<EnvironmentLight>(glm::vec3(0.8f));
}

void Scene::load(string const& path, glm::vec3 translate, glm::vec3 scale)
{
    // read file via ASSIMP
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_RemoveComponent | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);

    // remove extension from filename
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_name = base_filename.substr(0, p);

    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
        return;
    }
    // retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));

    if (scene->HasMeshes()) {
        aiMatrix4x4 model_mat = scene->mRootNode->mTransformation;
        load_model(model_mat, scene->mRootNode, scene);
    }

    if (scene->HasLights()) {
        for (unsigned int i = 0; i < scene->mNumLights; i++) {
            aiLight* light = scene->mLights[i];
            load_light(light, scene);
        }
    }

    if (scene->HasCameras()) {
        for (unsigned int i = 0; i < scene->mNumCameras; i++) {
            aiCamera* camera = scene->mCameras[i];
            load_camera(camera, scene);
        }
    }

    transfer_data_to_device();
}

void Scene::add_render_object(std::shared_ptr<RenderObject> r_obj)
{
    int id = r_obj->get_id();

    if (render_objects.find(id) == render_objects.end()) {

        render_objects.insert(std::pair<int, std::shared_ptr<RenderObject>>(id, r_obj));

        auto mat = r_obj->get_material();
        if (materials_loaded.find(mat->get_name()) == materials_loaded.end()) {
            materials_loaded.insert(std::pair<string, std::shared_ptr<Material>>(mat->get_name(), mat));
        }
    }
}

void Scene::remove_render_object(std::shared_ptr<RenderObject> r_obj)
{
    int id = r_obj->get_id();

    if (render_objects.find(id) != render_objects.end()) {
        render_objects.erase(id);
    }
}

void Scene::clear_render_objects()
{
    for (auto obj : render_objects) {
        remove_render_object(obj.second);
    }
}

void Scene::add_material(Material material)
{
    std::shared_ptr<Material> ptr_mat = std::shared_ptr<Material>(&material);
    materials_loaded.insert(std::pair<string, std::shared_ptr<Material>>(ptr_mat->get_name(), ptr_mat));
}

void Scene::add_material(std::shared_ptr<Material> material)
{
    materials_loaded.insert(std::pair<string, std::shared_ptr<Material>>(material->get_name(), material));
}

void Scene::remove_material(Material material)
{
    auto key = material.get_name();

    if (materials_loaded.find(key) != materials_loaded.end()) {
        auto mat = materials_loaded[key];
        materials_loaded.erase(key);
    }
}

void Scene::remove_material(std::shared_ptr<Material> material)
{
    auto key = material->get_name();

    if (materials_loaded.find(key) != materials_loaded.end()) {
        auto mat = materials_loaded[key];
        materials_loaded.erase(key);
    }
}

void Scene::remove_material(std::string mat_name)
{
    auto key = mat_name;

    if (materials_loaded.find(key) != materials_loaded.end()) {

        auto mat = materials_loaded[key];
        materials_loaded.erase(key);
    }
}

void Scene::add_light(std::shared_ptr<PointLight> light)
{
    if (find(point_lights.begin(), point_lights.end(), light) == point_lights.end()) {
        point_lights.push_back(light);
        light->attach(shared_from_this());
    }
}

void Scene::add_light(std::shared_ptr<DirectionalLight> light)
{
    if (find(dir_lights.begin(), dir_lights.end(), light) == dir_lights.end()) {
        dir_lights.push_back(light);
        light->attach(shared_from_this());
    }
}

void Scene::set_environment_light(std::shared_ptr<EnvironmentLight> environment_light)
{
    if ((this)->environment_light != nullptr)
    {
        (this)->environment_light->detach(shared_from_this());
    }

    (this)->environment_light = environment_light;
    environment_light->attach(shared_from_this());
}

void Scene::remove_light(std::shared_ptr<PointLight> light)
{
    auto light_it = find(point_lights.begin(), point_lights.end(), light);

    if (light_it != point_lights.end()) {
        point_lights.erase(light_it);
        light->detach(shared_from_this());
    }
}

void Scene::remove_light(std::shared_ptr<DirectionalLight> light)
{
    auto light_it = find(dir_lights.begin(), dir_lights.end(), light);

    if (light_it != dir_lights.end()) {
        dir_lights.erase(light_it);
        light->detach(shared_from_this());
    }
}

void Scene::load_model(aiMatrix4x4 accTransform, aiNode* node, const aiScene* scene)
{
    accTransform = node->mTransformation * accTransform;

    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        add_render_object(load_mesh(mesh, accTransform, scene));
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        load_model(accTransform, node->mChildren[i], scene);
    }
}

Light* Scene::load_light(aiLight* light, const aiScene* scene)
{
    return nullptr;
}

std::shared_ptr<Mesh> Scene::load_mesh(aiMesh* mesh, aiMatrix4x4 accTransform, const aiScene* scene)
{
    string name = mesh->mName.C_Str();

    // data to fill
    vector<Vertex> vertices;
    vector<unsigned int> indices;

    glm::mat4 mat = glm::transpose(glm::inverse(mat4_cast(accTransform)));

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 vector;

        // positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.position = glm::vec3(mat4_cast(accTransform) * glm::vec4(vector, 1));

        // normals
        if (mesh->HasNormals()) {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;

            vertex.normal = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));
        }

        // texture coordinates
        if (mesh->mTextureCoords[0]) {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.texCoords = vec;

            // tangent
            vector.x = mesh->mTangents[i].x;
            vector.y = mesh->mTangents[i].y;
            vector.z = mesh->mTangents[i].z;
            vertex.tangent = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));

            // bitangent
            vector.x = mesh->mBitangents[i].x;
            vector.y = mesh->mBitangents[i].y;
            vector.z = mesh->mBitangents[i].z;
            vertex.bitangent = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));
        }
        else {
            vertex.texCoords = glm::vec2(0.0f, 0.0f);
        }

        vertices.push_back(vertex);
    }

    // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];

        // retrieve all indices of the face and store them in the indices vector
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    std::shared_ptr<Mesh> mesh_out = std::make_shared<Mesh>(vertices, indices, load_material(material, scene));
    mesh_out->set_name(name);
    return mesh_out;
}

std::shared_ptr<Material> Scene::load_material(aiMaterial* material, const aiScene* scene)
{
    aiString name;
    material->Get(AI_MATKEY_NAME, name);

    if (name.length == 0) name = aiString(to_string((int)materials_loaded.size()));

    auto it = materials_loaded.find(name.C_Str());

    if (it == materials_loaded.end()) {
        std::shared_ptr<Material> mat = std::make_shared<Material>();

        mat->set_name(name.C_Str());

        //material->Get(AI_MATKEY_TWOSIDED, mat->doubleSided);

        aiColor3D base_color_factor, emissive_color_factor;
        ai_real roughness_factor, metallic_factor;

        material->Get(AI_MATKEY_COLOR_DIFFUSE, base_color_factor);
        material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color_factor);
        material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness_factor);
        material->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor);

        mat->set_base_color_factor(glm::make_vec3(&base_color_factor.r));
        mat->set_emissive_color_factor(glm::make_vec3(&emissive_color_factor.r));

        mat->set_roughness_factor(1.f);
        mat->set_metallic_factor(0.f);

        mat->set_base_color_texture(load_texture(aiTextureType_DIFFUSE, "base_color_texture", material, scene));
        mat->set_normal_texture(load_texture(aiTextureType_NORMALS, "normal_texture", material, scene));
        mat->set_occlusion_texture(load_texture(aiTextureType_AMBIENT_OCCLUSION, "occlusion_texture", material, scene));
        mat->set_emissive_texture(load_texture(aiTextureType_EMISSIVE, "emissive_texture", material, scene));
        mat->set_roughness_texture(load_texture(aiTextureType_DIFFUSE_ROUGHNESS, "roughness_texture", material, scene));
        mat->set_metallic_texture(load_texture(aiTextureType_METALNESS, "metallic_texture", material, scene));
        mat->set_metallic_roughness_texture(load_texture(aiTextureType_UNKNOWN, "metallic_roughness_texture", material, scene));

        add_material(mat);

        return mat;
    }
    else {
        return materials_loaded[name.C_Str()];
    }
}

std::shared_ptr<Texture> Scene::load_texture(aiTextureType type, string typeName, aiMaterial* mat, const aiScene* scene)
{
    std::shared_ptr<Texture> texture = nullptr;

    if (mat->GetTextureCount(type)) {
        aiString str;
        mat->GetTexture(type, 0, &str);

        // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
        bool skip = false;
        for (unsigned int j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j]->filepath.data(), str.C_Str()) == 0)
            {
                texture = textures_loaded[j];
            }
        }
        if (!skip)
        {   // if texture hasn't been loaded already, load it
            if ('*' == *str.C_Str()) {
                texture = std::make_shared<Texture>(scene->GetEmbeddedTexture(str.C_Str()), GL_TEXTURE_2D, 0, 0, 0, typeName);
            }
            else {
                string path = this->directory + '/' + str.C_Str();
                texture = std::make_shared<Texture>(path, GL_TEXTURE_2D, 0, 0, 0, typeName);
            }
            textures_loaded.push_back(texture);
        }
    }
    return texture;
}

std::shared_ptr<Camera> Scene::load_camera(aiCamera* camera, const aiScene* scene)
{
    return nullptr;
}

void Scene::transfer_data_to_device()
{
    checkCudaErrors(cudaMallocManaged(&dptr, sizeof(dScene)));

    thrust::device_vector<dLight*> all_lights;
    all_lights.reserve(dir_lights.size() + point_lights.size() + 1);
    
    all_lights.push_back(environment_light->dptr());

    for (auto const& l : dir_lights)
    {
        all_lights.push_back(l->dptr());
    }
    for (auto const& l : point_lights)
    {
        all_lights.push_back(l->dptr());
    }

    //dptr->lights = thrust::raw_pointer_cast(all_lights.data());

    size_t light_size = sizeof(dLight*) * all_lights.size();
    checkCudaErrors(cudaMalloc(&dptr->lights, light_size));
    checkCudaErrors(cudaMemcpy(dptr->lights, thrust::raw_pointer_cast(all_lights.data()), light_size, cudaMemcpyDeviceToDevice));

    dptr->environment_light = static_cast<dEnvironmentLight*>(environment_light->dptr());
    dptr->nmb_lights = all_lights.size();

    d_build_bvh();
}

void Scene::create_dptr()
{
}
void Scene::delete_dptr()
{
}

void Scene::d_build_bvh()
{
#ifdef LOG
    std::cerr << "\t gathering triangle information for BVH construction: ";
    clock_t start, stop;
    start = clock();
#endif

    thrust::device_vector<dTriangle> d_triangles;
    std::vector<BVHPrimitiveInfo> bvh_tri_info;

    d_init_triangle_data(d_triangles);
    d_init_bvh_triangle_info(d_triangles, bvh_tri_info);

    bvh = new BVHAccel(bvh_tri_info, SplitMethod::SAH, 8);

    d_reorder_triangles(bvh->get_ordered_prims(), d_triangles);

    //dptr->triangles = thrust::raw_pointer_cast(d_triangles.data());
    dptr->nmb_triangles = d_triangles.size();

    size_t tri_size = sizeof(dTriangle) * d_triangles.size();
    checkCudaErrors(cudaMalloc(&dptr->triangles, tri_size));
    checkCudaErrors(cudaMemcpy(dptr->triangles, thrust::raw_pointer_cast(d_triangles.data()), tri_size, cudaMemcpyDeviceToDevice));

    size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    checkCudaErrors(cudaMalloc(&dptr->nodes, node_size));
    checkCudaErrors(cudaMemcpy(dptr->nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));

#ifdef LOG
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "s" << endl;
#endif
}
void Scene::d_init_triangle_data(thrust::device_vector<dTriangle>& d_triangles)
{
    auto nmb_triangles = 0;
    for (auto const& r_obj : render_objects)
    {
        auto mesh = std::dynamic_pointer_cast<Mesh>(r_obj.second);
        if (mesh != nullptr)
        {
            auto d_triangles_i = mesh->get_device_triangles();
            nmb_triangles += d_triangles_i.size();

            d_triangles.reserve(nmb_triangles);
            d_triangles.insert(d_triangles.end(), d_triangles_i.begin(), d_triangles_i.end());
        }
    }
}
void Scene::d_init_bvh_triangle_info(const thrust::device_vector<dTriangle>& d_triangles, std::vector<BVHPrimitiveInfo>& bvh_tri_info)
{
    thrust::device_vector<BVHPrimitiveInfo> d_bvh_tri_info(d_triangles.size());
    init_BVH_triangle_info(d_triangles.size(), thrust::raw_pointer_cast(d_triangles.data()), thrust::raw_pointer_cast(d_bvh_tri_info.data()));
    
    bvh_tri_info.resize(d_triangles.size());
    thrust::copy(d_bvh_tri_info.begin(), d_bvh_tri_info.end(), bvh_tri_info.begin());
}
void Scene::d_reorder_triangles(const thrust::device_vector<int>& d_reorder_map, thrust::device_vector<dTriangle>& d_triangles)
{
    thrust::device_vector<dTriangle> d_output(d_triangles.size());
    thrust::gather(
        d_reorder_map.begin(),
        d_reorder_map.end(),
        d_triangles.begin(),
        d_output.begin());
    d_triangles.swap(d_output);
    //d_output.clear();
}

/*
void Scene::build_BVH()
{
#ifdef LOG
    std::cerr << "\t building BVH: ";
    clock_t start, stop;
    start = clock();
#endif
    bvh = new BVHAccel(BVH_triangle_info, nmb_triangles, SplitMethod::SAH, 8);
#ifdef LOG
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "s" << endl;
#endif
}
*/

void Scene::send_uniforms(Shader& shader)
{
    for (unsigned int i = 0; i < point_lights.size(); i++)
    {
        shader.setVec3("pnt_lights[" + std::to_string(i) + "].position", point_lights[i]->getPosition());
        shader.setVec3("pnt_lights[" + std::to_string(i) + "].color", point_lights[i]->get_color());
        shader.setFloat("pnt_lights[" + std::to_string(i) + "].intensity", point_lights[i]->get_ls());
    }

    for (unsigned int i = 0; i < dir_lights.size(); i++)
    {
        shader.setVec3("dir_lights[" + std::to_string(i) + "].direction", dir_lights[i]->get_dir());
        shader.setVec3("dir_lights[" + std::to_string(i) + "].color", dir_lights[i]->get_color());
        shader.setFloat("dir_lights[" + std::to_string(i) + "].intensity", dir_lights[i]->get_ls());
    }

    // TODO: cone lights at some point
}

void Scene::bind_environment_textures(Shader& shader)
{
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_CUBE_MAP, environment_light->getIrradianceMapID());

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_CUBE_MAP, environment_light->getPrefilterMapID());

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, environment_light->get_brdfLUT_ID());
}

dScene* Scene::get_dptr()
{
    return dptr;
}

vector<std::shared_ptr<DirectionalLight>> Scene::get_lights()
{
    return dir_lights;
}

std::shared_ptr<EnvironmentLight> Scene::get_environment_light()
{
    return environment_light;
}

void Scene::notify(const std::string& msg)
{
    for (auto const& o : observers)
    {
        o->update("");
    }
}

void Scene::update(const std::string& msg)
{
    notify(msg);
}

/*
dCamera* Scene::get_camera() const
{
    return dscene->get_camera();
}

LinearBVHNode* Scene::get_nodes() const
{
    return dscene->get_nodes();
}

dTriangle* Scene::get_triangles() const
{
    return dscene->get_triangles();
}

dEnvironmentLight* Scene::get_environment_light() const
{
    return dscene->get_environment_light();
}

dLight** Scene::get_lights() const
{
    return dscene->get_lights();
}
*/