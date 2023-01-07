#include "Material.h"
#include "globals.h"

#include <glm/gtx/string_cast.hpp>

#include "CudaHelpers.h"
#include "dTexture.cuh"

Material::Material()
{
	id = gen_id();
	name = gen_object_name("Mat");

	alphaMode = 0.f;
	alphaCutoff = 0.f;

	doubleSided = false;

	base_color_factor = glm::vec3(1.f);
	emissive_color_factor = glm::vec3(0.f);
	fresnel = glm::vec3(0.04f);

	roughness_factor = 1.f;
	metallic_factor = 0.f;

	create_dptr();
}
Material::Material(const Material& mat)
{
	name = gen_object_name(mat.name + " Copy");

	alphaMode = mat.alphaMode;
	alphaCutoff = mat.alphaCutoff;

	doubleSided = mat.doubleSided;

	base_color_factor = mat.base_color_factor;
	emissive_color_factor = mat.emissive_color_factor;
	fresnel = mat.fresnel;

	roughness_factor = mat.roughness_factor;
	metallic_factor = mat.metallic_factor;

	base_color_texture = mat.base_color_texture;
	normal_texture = mat.normal_texture;
	occlusion_texture = mat.occlusion_texture;
	emissive_texture = mat.emissive_texture;
	roughness_texture = mat.roughness_texture;
	metallic_texture = mat.metallic_texture;
	metallic_roughness_texture = mat.metallic_roughness_texture;

	create_dptr();
}

Material::~Material()
{
}

void Material::send_uniforms(const Shader& shader)
{
	shader.setVec3("base_color_factor", base_color_factor);
	shader.setVec3("emissive_color_factor", emissive_color_factor);

	shader.setFloat("roughness_factor", roughness_factor);
	shader.setFloat("metallic_factor", metallic_factor);

	shader.setBool("base_color_texture_sample", (base_color_texture != nullptr));
	shader.setBool("normal_texture_sample", (normal_texture != nullptr));
	shader.setBool("occlusion_texture_sample", (occlusion_texture != nullptr));
	shader.setBool("emissive_texture_sample", (emissive_texture != nullptr));
	shader.setBool("metallic_roughness_texture_sample", (metallic_roughness_texture != nullptr));
	shader.setBool("roughness_texture_sample", (roughness_texture != nullptr));
	shader.setBool("metallic_texture_sample", (metallic_texture != nullptr));

	shader.setInt("base_color_texture", 0);
	shader.setInt("normal_texture", 1);
	shader.setInt("occlusion_texture", 2);
	shader.setInt("emissive_texture", 3);
	shader.setInt("metallic_roughness_texture", 4);
	shader.setInt("roughness_texture", 5);
	shader.setInt("metallic_texture", 6);

	if (base_color_texture != nullptr) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, base_color_texture->id);
	}

	if (normal_texture != nullptr) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, normal_texture->id);
	}

	if (occlusion_texture != nullptr) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, occlusion_texture->id);
	}

	if (emissive_texture != nullptr) {
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, emissive_texture->id);
	}

	if (metallic_roughness_texture != nullptr) {
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, metallic_roughness_texture->id);
	}

	if (roughness_texture != nullptr) {
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, roughness_texture->id);
	}

	if (metallic_texture != nullptr) {
		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D, metallic_texture->id);
	}
}

int Material::get_id() const
{
	return id;
}
std::string Material::get_name() const
{
	return name;
}

glm::vec3 Material::get_base_color_factor() const
{
	return base_color_factor;
}
glm::vec3 Material::get_emissive_color_factor() const
{
	return emissive_color_factor;
}
glm::vec3 Material::get_fresnel() const
{
	return fresnel;
}
float Material::get_roughness_factor() const
{
	return roughness_factor;
}
float Material::get_metallic_factor() const
{
	return metallic_factor;
}

std::shared_ptr<Texture> Material::get_base_color_texture() const
{
	return base_color_texture;
}
std::shared_ptr<Texture> Material::get_normal_texture() const
{
	return normal_texture;
}
std::shared_ptr<Texture> Material::get_occlusion_texture() const
{
	return occlusion_texture;
}
std::shared_ptr<Texture> Material::get_emissive_texture() const
{
	return emissive_texture;
}
std::shared_ptr<Texture> Material::get_roughness_texture() const
{
	return roughness_texture;
}
std::shared_ptr<Texture> Material::get_metallic_texture() const
{
	return metallic_texture;
}
std::shared_ptr<Texture> Material::get_metallic_roughness_texture() const
{
	return metallic_roughness_texture;
}

dMaterial* Material::get_dptr() const
{
	return dptr;
}

void Material::set_name(const ::string& name)
{
	(this)->name = name;
}

void Material::set_base_color_factor(const glm::vec3& base_color_factor)
{
	(this)->base_color_factor = base_color_factor;
	(this)->dptr->base_color_factor = base_color_factor;
}
void Material::set_emissive_color_factor(const glm::vec3& emissive_color_factor)
{
	(this)->emissive_color_factor = emissive_color_factor;
	(this)->dptr->emissive_color_factor = emissive_color_factor;
}
void Material::set_fresnel(const glm::vec3& fresnel)
{
	(this)->fresnel = fresnel;
	(this)->dptr->fresnel = fresnel;
}
void Material::set_roughness_factor(const float roughness_factor)
{
	(this)->roughness_factor = roughness_factor;
	(this)->dptr->roughness_factor = roughness_factor;
}
void Material::set_metallic_factor(const float metallic_factor)
{
	(this)->metallic_factor = metallic_factor;
	(this)->dptr->metallic_factor = metallic_factor;
}

void Material::set_base_color_texture(const std::shared_ptr<Texture> base_color_texture)
{
	(this)->base_color_texture = base_color_texture;
	(this)->dptr->base_color_texture = load_texture_uchar(base_color_texture.get());
}
void Material::set_normal_texture(const std::shared_ptr<Texture> normal_texture)
{
	(this)->normal_texture = normal_texture;
	(this)->dptr->normal_texture = load_texture_uchar(normal_texture.get());
}
void Material::set_occlusion_texture(const std::shared_ptr<Texture> occlusion_texture)
{
	(this)->occlusion_texture = occlusion_texture;
	(this)->dptr->occlusion_texture = load_texture_uchar(occlusion_texture.get());
}
void Material::set_emissive_texture(const std::shared_ptr<Texture> emissive_texture)
{
	(this)->emissive_texture = emissive_texture;
	(this)->dptr->emissive_texture = load_texture_uchar(emissive_texture.get());
}
void Material::set_roughness_texture(const std::shared_ptr<Texture> roughness_texture)
{
	(this)->roughness_texture = roughness_texture;
	(this)->dptr->roughness_texture = load_texture_uchar(roughness_texture.get());
}
void Material::set_metallic_texture(const std::shared_ptr<Texture> metallic_texture)
{
	(this)->metallic_texture = metallic_texture;
	(this)->dptr->metallic_texture = load_texture_uchar(metallic_texture.get());
}
void Material::set_metallic_roughness_texture(const std::shared_ptr<Texture> metallic_roughness_texture)
{
	(this)->metallic_roughness_texture = metallic_roughness_texture;
	(this)->dptr->metallic_roughness_texture = load_texture_uchar(metallic_roughness_texture.get());
}

void Material::create_dptr()
{
	checkCudaErrors(cudaMallocManaged(&dptr, sizeof(dMaterial)));
	dptr->base_color_factor = base_color_factor;
	dptr->roughness_factor = roughness_factor;
	dptr->metallic_factor = metallic_factor;
	dptr->emissive_color_factor = emissive_color_factor;
	dptr->fresnel = fresnel;
	dptr->radiance = 0.f;
	dptr->emissive = false;
}
void Material::delete_dptr()
{
	if (dptr != nullptr) {
		checkCudaErrors(cudaFree(dptr));
		dptr = nullptr;
	}
}
