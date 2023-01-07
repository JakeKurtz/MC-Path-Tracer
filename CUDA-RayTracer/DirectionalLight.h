#pragma once
#include "Light.h"

class Isect;

struct dDirectionalLight : public dLight
{
	jek::Vec3f dir;
private:
	__device__ void compute_dir(const Isect& isect, jek::Vec3f& wi) const;
	__device__ jek::Vec3f compute_L(const Isect& isect, const jek::Vec3f& wi) const;
	__device__ jek::Vec3f compute_L(const Isect& isect) const;
	__device__ float compute_pdf(const Isect& isect, const jek::Vec3f& wi) const;
	__device__ float compute_pdf(const Isect& isect) const;
};

class DirectionalLight : public Light
{
private:
	glm::vec3 dir;
	void init_dptr();
	dDirectionalLight* get_dptr() const;
public:
	DirectionalLight();
	DirectionalLight(const glm::vec3 dir, const glm::vec3 color);
	~DirectionalLight();
	void set_dir(const glm::vec3 _direction);
	glm::vec3 get_dir() const;
};

