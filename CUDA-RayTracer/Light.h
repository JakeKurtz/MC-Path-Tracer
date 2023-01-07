#pragma once

#include "GLCommon.h"
#include "Subject.h"
#include <string>

#include <cuda_runtime.h>

#include <Vector.h>

class dRay;
class Isect;
class LinearBVHNode;
class dTriangle;

struct dLight
{
	__device__ bool is_delta() const;
	__device__ bool visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const;
	__device__ void dir(const Isect& isect, jek::Vec3f& wi) const;
	__device__ jek::Vec3f L(const Isect& isect, const jek::Vec3f& wi) const;
	__device__ jek::Vec3f L(const Isect& isect) const;
	__device__ float pdf(const Isect& isect, const jek::Vec3f& wi) const;
	__device__ float pdf(const Isect& isect) const;

	jek::Vec3f color = jek::Vec3f(1.f, 1.f, 1.f);
	float ls = 1.f;
	bool enable_shadows;
	bool delta = false;

private:
	__device__ virtual void compute_dir(const Isect& isect, jek::Vec3f& wi) const = 0;
	__device__ virtual jek::Vec3f compute_L(const Isect& isect, const jek::Vec3f& wi) const = 0;
	__device__ virtual jek::Vec3f compute_L(const Isect& isect) const = 0;
	__device__ virtual float compute_pdf(const Isect& isect, const jek::Vec3f& wi) const = 0;
	__device__ virtual float compute_pdf(const Isect& isect) const = 0;
};

class Light : public Subject
{
public:
	std::string get_name() const;
	int get_id() const;
	glm::vec3 get_color() const;
	float get_ls() const;
	float get_range() const;

	dLight* dptr() const;

	void set_color(const glm::vec3 color);
	void set_ls(const float ls);
	void set_range(const float range);

protected:
	std::string name;
	int id;

	bool delta = false;

	glm::vec3 color;
	float ls = 1.f;
	float range = 100.f;

	dLight* d_light;

	void notify(const std::string& msg);

private:
	virtual void init_dptr() = 0;
	virtual dLight* get_dptr() const = 0;
};

