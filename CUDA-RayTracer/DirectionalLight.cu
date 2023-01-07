#include "DirectionalLight.h"
#include "globals.h"
#include "CudaHelpers.h"
#include "Isect.cuh"
#include "light_initialization_kernels.cuh"
#include <Sample.h>

__device__ void dDirectionalLight::compute_dir(const Isect& isect, jek::Vec3f& wi) const
{
	wi = dir;
}
__device__ jek::Vec3f dDirectionalLight::compute_L(const Isect& isect, const jek::Vec3f& wi) const
{
	/*
	auto center = dir * 10000.f;
	auto orig = jek::Vec3f(0.f);
	auto r = 100.f;
	auto r2 = r * r;
	jek::Vec3f L = center - orig;
	float tca = jek::dot(L,wi);
	float d2 = jek::dot(L,L) - tca * tca;
	if (d2 > r2) return jek::Vec3f(0.f);
	float thc = sqrt(r2 - d2);
	float t0 = tca - thc;
	float t1 = tca + thc;

	if (t0 > t1) { float c(t0); t0 = t1; t1 = c; }

	if (t0 < 0) {
		t0 = t1;
		if (t0 < 0) return jek::Vec3f(0.f);
	}
	*/
	return ls * color;
}
__device__ jek::Vec3f dDirectionalLight::compute_L(const Isect& isect) const
{
	return ls * color;
}
__device__ float dDirectionalLight::compute_pdf(const Isect& isect, const jek::Vec3f& wi) const
{
	return 1.f;
}
__device__ float dDirectionalLight::compute_pdf(const Isect& isect) const
{
	return 1.f;
}

DirectionalLight::DirectionalLight()
{
	(this)->id = gen_id();
	(this)->name = gen_object_name("Directional Light");
	(this)->dir = glm::vec3(1.f);
	(this)->color = glm::vec3(1.f);
	init_dptr();
}
DirectionalLight::DirectionalLight(glm::vec3 dir, glm::vec3 color)
{
	(this)->id = gen_id();
	(this)->name = gen_object_name("Directional Light");
	(this)->dir = dir;
	(this)->color = color;
	init_dptr();
}
DirectionalLight::~DirectionalLight()
{
	cudaFree(d_light);
	remove_object_name(name);
}
void DirectionalLight::init_dptr()
{
	checkCudaErrors(cudaMallocManaged(&d_light, sizeof(dDirectionalLight)));

	auto _d_light = static_cast<dDirectionalLight*>(d_light);
	init_light_on_device(_d_light);

	_d_light->color = color;
	_d_light->dir = dir;
	_d_light->ls = ls;
	_d_light->delta = true;
}
void DirectionalLight::set_dir(const glm::vec3 dir) {
	(this)->dir = glm::normalize(dir);

	auto _d_light = static_cast<dDirectionalLight*>(d_light);
	_d_light->dir = dir;
	notify("");
}
glm::vec3 DirectionalLight::get_dir() const
{ 
	return dir; 
}
dDirectionalLight* DirectionalLight::get_dptr() const
{
	auto _d_light = static_cast<dDirectionalLight*>(d_light);
	return _d_light;
}
