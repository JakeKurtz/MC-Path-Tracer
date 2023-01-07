#pragma once

#include "GLCommon.h"

#include "Texture.h"
#include "Shader.h"
#include "dMaterial.cuh"

#include <string>

class Material
{
public:
	Material();
	Material(const Material& mat);
	~Material();
	void send_uniforms(const Shader& shader);

	int get_id() const;
	std::string get_name() const;

	glm::vec3 get_base_color_factor() const;
	glm::vec3 get_emissive_color_factor() const;
	glm::vec3 get_fresnel() const;

	float get_roughness_factor() const;
	float get_metallic_factor() const;

	std::shared_ptr<Texture> get_base_color_texture() const;
	std::shared_ptr<Texture> get_normal_texture() const;
	std::shared_ptr<Texture> get_occlusion_texture() const;
	std::shared_ptr<Texture> get_emissive_texture() const;
	std::shared_ptr<Texture> get_roughness_texture() const;
	std::shared_ptr<Texture> get_metallic_texture()const;
	std::shared_ptr<Texture> get_metallic_roughness_texture() const;

	dMaterial* get_dptr() const;

	void set_name(const std::string& name);

	void set_base_color_factor(const glm::vec3& base_color_factor);
	void set_emissive_color_factor(const glm::vec3& emissive_color_factor);
	void set_fresnel(const glm::vec3& fresnel);

	void set_roughness_factor(const float roughness_factor);
	void set_metallic_factor(const float metallic_factor);

	void set_base_color_texture(const std::shared_ptr<Texture> base_color_texture);
	void set_normal_texture(const std::shared_ptr<Texture> normal_texture);
	void set_occlusion_texture(const std::shared_ptr<Texture> occlusion_texture);
	void set_emissive_texture(const std::shared_ptr<Texture> emissive_texture);
	void set_roughness_texture(const std::shared_ptr<Texture> roughness_texture);
	void set_metallic_texture(const std::shared_ptr<Texture> metallic_texture);
	void set_metallic_roughness_texture(const std::shared_ptr<Texture> metallic_roughness_texture);

private:
	int id;
	std::string name;

	dMaterial* dptr;

	int alphaMode;
	float alphaCutoff;

	bool doubleSided;

	glm::vec3 base_color_factor;
	glm::vec3 emissive_color_factor;
	glm::vec3 fresnel;

	float roughness_factor = 1.f;
	float metallic_factor = 0.f;

	std::shared_ptr<Texture> base_color_texture = nullptr;
	std::shared_ptr<Texture> normal_texture = nullptr;
	std::shared_ptr<Texture> occlusion_texture = nullptr;
	std::shared_ptr<Texture> emissive_texture = nullptr;
	std::shared_ptr<Texture> roughness_texture = nullptr;
	std::shared_ptr<Texture> metallic_texture = nullptr;
	std::shared_ptr<Texture> metallic_roughness_texture = nullptr;

	void create_dptr();
	void delete_dptr();
};

