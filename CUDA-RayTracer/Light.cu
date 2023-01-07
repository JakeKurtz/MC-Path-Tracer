#include "Light.h"

#include "Isect.cuh"
#include "dRay.cuh"
#include "Triangle.h"
#include "BVH.h"

__device__ bool dLight::is_delta() const
{
	return delta;
}
__device__ bool dLight::visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const
{
	float ts = jek::K_HUGE;
	return !intersect_shadows(nodes, triangles, ray, ts);
}
__device__ void dLight::dir(const Isect& isect, jek::Vec3f& wi) const
{
	return compute_dir(isect, wi);
}
__device__ jek::Vec3f dLight::L(const Isect& isect, const jek::Vec3f& wi) const
{
	return compute_L(isect, wi);
}
__device__ jek::Vec3f dLight::L(const Isect& isect) const
{
	return compute_L(isect);
}
__device__ float dLight::pdf(const Isect& isect, const jek::Vec3f& wi) const
{
	return compute_pdf(isect, wi);
}
__device__ float dLight::pdf(const Isect& isect) const
{
	return compute_pdf(isect);
}

std::string Light::get_name() const
{
	return name;
}
int Light::get_id() const
{ 
	return id; 
}
glm::vec3 Light::get_color() const
{ 
	return color; 
}
float Light::get_ls() const
{ 
	return ls; 
}
float Light::get_range() const
{ 
	return range; 
}
dLight* Light::dptr() const
{
	return get_dptr();
}
void Light::set_color(const glm::vec3 color)
{ 
	(this)->color = color; 
	(this)->d_light->color = color;
	notify("");
}
void Light::set_ls(const float ls)
{
	(this)->ls = ls; 
	(this)->d_light->ls = ls;
	notify("");
}
void Light::set_range(const float range)
{ 
	(this)->range = range; 
	notify("");
}
void Light::notify(const std::string& msg)
{
	for (auto const& o : observers)
    {
        o->update(msg);
    }
}