/*
#include "dAreaLight.cuh"
#include "GeometricObj.cuh"
#include "Isect.cuh"
#include "dRay.cuh"
#include "dTriangle.cuh"

__device__ dAreaLight::dAreaLight(void) :
	dLight(),
	object_ptr(nullptr)//,
//	material(nullptr)
{
	ls = 1.f;
	position = make_float3(0, 0, 0);
	color = make_float3(1, 1, 1);
	delta = false;
}

__device__ void dAreaLight::get_direction(const Isect isect, float3& wi, float3& sample_point) const
{
	sample_point = object_ptr->sample();
	wi = normalize(sample_point - isect.position);
};

__device__ float3 dAreaLight::L(const Isect isect, const float3 wi, const float3 sample_point) const
{
	float3 light_normal = object_ptr->get_normal(sample_point);
	float n_dot_d = dot(-light_normal, wi);

	if (n_dot_d > 0.0)
		return (emissive_L(material));
	else
		return (make_float3(0.f));
};

__device__ float dAreaLight::get_pdf(const Isect isect) const
{
	float3 sample_point = object_ptr->sample();
	float3 light_normal = object_ptr->get_normal(sample_point);
	float3 wi = normalize(sample_point - isect.position);

	float n_dot_d = abs(dot(light_normal, -wi));
	float d2 = pow(distance(sample_point, isect.position), 2);

	return ((d2 / n_dot_d) * object_ptr->pdf(isect));
}

__device__ float dAreaLight::get_pdf(const Isect isect, const float3 wi) const
{
	float3 sample_point = object_ptr->sample();
	float3 light_normal = object_ptr->get_normal(sample_point);
	float3 _wi = normalize(sample_point - isect.position);

	float n_dot_d = abs(dot(light_normal, -_wi));
	float d2 = pow(distance(sample_point, isect.position), 2);

	return ((d2 / n_dot_d) * object_ptr->pdf(isect));
};

__device__ bool dAreaLight::visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay ray) const
{
	/*if (!object_ptr->hit(ray)) {
		return false;
	}
	else {
		float ts = dot((object_ptr->sample() - ray.o), ray.d);
		return !intersect_shadows(nodes, triangles, ray, ts) ? 1.f : 0.f;;
	}
	*/
//	return true;
//};
/*
__device__ void dAreaLight::set_position(const float x, const float y, const float z)
{
	position = make_float3(x, y, z);
};

__device__ void dAreaLight::set_position(const float3 pos)
{
	position = pos;
}

__device__ void dAreaLight::set_object(GeometricObj* obj_ptr)
{
	object_ptr = obj_ptr;
};
*/