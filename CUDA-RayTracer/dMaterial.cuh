#pragma once
#include <cuda_runtime.h>

#include "Vector.h"

class Isect;
class dLight;
class LinearBVHNode;
class dTriangle;

struct dMaterial
{
	jek::Vec3f base_color_factor = jek::Vec3f(1.f, 1.f, 1.f);
	jek::Vec3f emissive_color_factor = jek::Vec3f(0.f, 0.f, 0.f);
	jek::Vec3f fresnel = jek::Vec3f(0.04f, 0.04f, 0.04f);

	float roughness_factor = 1.f;
	float metallic_factor = 0.f;

	float ks = 1.f;
	float kd = 1.f;
	float radiance = 0.f;

	bool emissive = 0;

	int base_color_texture = -1;
	int normal_texture = -1;
	int occlusion_texture = -1;
	int emissive_texture = -1;
	int roughness_texture = -1;
	int metallic_texture = -1;
	int metallic_roughness_texture = -1;
};

__device__ jek::Vec3f refract(const jek::Vec3f& I, const jek::Vec3f& N, const float ior);
__device__ float power_heuristic(int nf, float fPdf, int ng, float gPdf);

__device__ jek::Vec3f get_albedo(const Isect& isect);
__device__ float get_roughness(const Isect& isect);
__device__ float get_metallic(const Isect& isect);
__device__ jek::Vec3f get_normal(const Isect& isect);

__device__ jek::Vec3f fresnel_schlick(const jek::Vec3f& f0, const jek::Vec3f& v, const jek::Vec3f& h);
__device__ jek::Vec3f fresnel_roughness(const jek::Vec3f& f0, const jek::Vec3f& n, const jek::Vec3f& wo, const float r);

__device__ float ndf_ggx_tr(const jek::Vec3f& n, const jek::Vec3f& h, const float r);
__device__ float ndf_beckmann(const jek::Vec3f& n, const jek::Vec3f& h, const float r);

__device__ float g1_ggx(const jek::Vec3f& v, const jek::Vec3f& n, const float r);
__device__ float g1_beckmann(const jek::Vec3f& v, const jek::Vec3f& n, const float r);
__device__ float g1_schlick_beckmann(const jek::Vec3f& v, const jek::Vec3f& n, const float r);
__device__ float g1_schlick_ggx(const jek::Vec3f& v, const jek::Vec3f& n, const float r);

__device__ float geo_atten_ggx(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r);
__device__ float geo_atten_beckmann(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r);
__device__ float geo_atten_schlick_beckmann(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r);
__device__ float geo_atten_schlick_ggx(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r);

__device__ jek::Vec3f diff_get_wi(const Isect& isect);
__device__ float diff_get_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);
__device__ jek::Vec3f diff_get_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);

__device__ jek::Vec3f spec_get_wi(const Isect& isect, const jek::Vec3f& wo);
__device__ float spec_get_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);
__device__ jek::Vec3f spec_get_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);

__device__ jek::Vec3f BRDF_L(const dLight* light, const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);
__device__ jek::Vec3f BRDF_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);
__device__ float BRDF_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo);

__device__ jek::Vec3f emissive_L(const Isect& isect, const jek::Vec3f& ray_dir);
__device__ jek::Vec3f emissive_L(const Isect& isect);
__device__ jek::Vec3f emissive_L(const dMaterial* material_ptr);