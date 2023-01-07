#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "GLCommon.h"

#include "Observer.h"

#include "Curve.h"
#include "Bezier.h"
#include "ClosedCurve.h"
#include "Mesh.h"
#include "Camera.h"
#include "EnvironmentLight.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "RenderObject.h"
#include "BVH.h"
#include "Subject.h"
#include "Observer.h"

struct dScene
{
	LinearBVHNode* nodes;
	dTriangle* triangles;
	dLight** lights;
	dEnvironmentLight* environment_light;

	uint32_t nmb_lights;
	uint32_t nmb_triangles;
};

class Scene : public std::enable_shared_from_this<Scene>, public Subject, public Observer
{
public:
	Scene();
	Scene(vector<std::shared_ptr<RenderObject>> models, vector<std::shared_ptr<Light>> lights, std::shared_ptr<Camera> camera);
	void load(string const& path, glm::vec3 translate = glm::vec3(0), glm::vec3 scale = glm::vec3(1));
	void add_render_object(std::shared_ptr<RenderObject> r_obj);
	void remove_render_object(std::shared_ptr<RenderObject> r_obj);
	void clear_render_objects();

	void add_material(Material material);
	void add_material(std::shared_ptr<Material> material);

	void remove_material(Material material);
	void remove_material(std::shared_ptr<Material> material);
	void remove_material(std::string mat_name);

	void add_light(std::shared_ptr<PointLight> light);
	void add_light(std::shared_ptr<DirectionalLight> light);

	void remove_light(std::shared_ptr<PointLight> light);
	void remove_light(std::shared_ptr<DirectionalLight> light);

	void set_environment_light(std::shared_ptr<EnvironmentLight> _environmentLight);

	void send_uniforms(Shader& shader);
	void bind_environment_textures(Shader& shader);

	dScene* get_dptr();

	vector<std::shared_ptr<DirectionalLight>> get_lights();
	std::shared_ptr<EnvironmentLight> get_environment_light();

	void notify(const std::string& msg);
	void update(const std::string& msg);

	vector<std::shared_ptr<PointLight>> point_lights;
	vector<std::shared_ptr<DirectionalLight>> dir_lights;
	std::shared_ptr<EnvironmentLight> environment_light;
	vector<std::shared_ptr<Texture>> textures_loaded;
	std::map<std::string, std::shared_ptr<Material>> materials_loaded;
	std::map<int, std::shared_ptr<RenderObject>> render_objects;

private:

	BVHAccel* bvh;
	dScene* dptr;

	string directory;

	void load_model(aiMatrix4x4 model_mat, aiNode* node, const aiScene* scene);
	Light* load_light(aiLight* light, const aiScene* scene);
	std::shared_ptr<Mesh> load_mesh(aiMesh* mesh, aiMatrix4x4 accTransform, const aiScene* scene);
	std::shared_ptr<Material> load_material(aiMaterial* material, const aiScene* scene);
	std::shared_ptr<Texture> load_texture(aiTextureType type, string typeName, aiMaterial* mat, const aiScene* scene);
	std::shared_ptr<Camera> load_camera(aiCamera* camera, const aiScene* scene);

	void transfer_data_to_device();
	void create_dptr();
	void delete_dptr();

	void d_build_bvh();
	void d_init_triangle_data(thrust::device_vector<dTriangle>& d_triangles);
	void d_init_bvh_triangle_info(const thrust::device_vector<dTriangle>& d_triangles, std::vector<BVHPrimitiveInfo>& d_bvh_tri_info);
	void d_reorder_triangles(const thrust::device_vector<int>& d_reorder_map, thrust::device_vector<dTriangle>& d_triangles);
};

