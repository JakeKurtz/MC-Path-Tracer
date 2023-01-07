#include "dMaterial.cuh"
#include "Isect.cuh"
#include "dMath.cuh"
#include "Light.h"
#include "dRay.cuh"
#include "Triangle.h"

__device__ const float epsilon = 0.00001f;

__device__ jek::Vec3f get_albedo(const Isect& isect) 
{
	int id = -1;
	if (isect.material != nullptr) {
		id = isect.material->base_color_texture;
	}

	jek::Vec3f albedo;
	if (id == -1) {
		albedo = isect.material->base_color_factor;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		albedo = tex2DLod<float4>(id, u, v, 0);
	}
	return isect.material->base_color_factor;
}
__device__ float get_roughness(const Isect& isect) 
{
	int mr_id = -1;
	int r_id = -1;
	if (isect.material != nullptr) {
		mr_id = isect.material->metallic_roughness_texture;
		r_id = isect.material->roughness_texture;
	}

	float r;
	if (mr_id != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		auto tex = tex2DLod<float4>(isect.material->metallic_roughness_texture, u, v, 0);
		r = fmax(tex.y, epsilon);
	}
	else if (r_id != -1 ){
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		auto tex = tex2DLod<float4>(isect.material->roughness_texture, u, v, 0);
		r = fmax(tex.x, epsilon);
	}
	else {
		r = fmax(isect.material->roughness_factor, epsilon);
	}

	return fmax(isect.material->roughness_factor, epsilon);
}
__device__ float get_metallic(const Isect& isect) 
{
	int mr_id = -1;
	int m_id = -1;
	if (isect.material != nullptr) {
		mr_id = isect.material->metallic_roughness_texture;
		m_id = isect.material->metallic_texture;
	}

	float m;
	if (mr_id != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		auto tex = tex2DLod<float4>(isect.material->metallic_roughness_texture, u, v, 0);
		m = tex.z;
	}
	else if (m_id != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		auto tex = tex2DLod<float4>(isect.material->metallic_texture, u, v, 0);
		m = tex.x;
	}
	else {
		m = isect.material->metallic_factor;
	}
	return isect.material->metallic_factor;
}
__device__ jek::Vec3f get_normal(const Isect& isect) 
{
	int id = -1;
	if (isect.material != nullptr) {
		id = isect.material->normal_texture;
	}

	jek::Vec3f normal;
	if (id == -1) {
		normal = isect.normal;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		
		jek::Vec3f T = isect.tangent;
		jek::Vec3f B = isect.bitangent;
		jek::Vec3f N = isect.normal;
		/*
		Matrix4x4 TBN = Matrix4x4(
			T.x, T.y, T.z, 0.f,
			B.x, B.y, B.z, 0.f,
			N.x, N.y, N.z, 0.f,
			0.f, 0.f, 0.f, 1.f
		);
		*/
		jek::Vec3f n = tex2DLod<float4>(id, u, v, 0);
		n = n * 2.f - 1.f;
		//normal = jek::normalize(TBN * jek::Vec3f(n.x, n.y, n.z));
		normal = jek::normalize(T * n.x + N * n.y + B * n.z);
	}
	return isect.normal;
}

__device__ jek::Vec3f refract(const jek::Vec3f& I, const jek::Vec3f& N, const float ior)
{
	float cosi = jek::clamp(dot(I, N), -1.f, 1.f);
	float etai = 1, etat = ior;
	jek::Vec3f n = N;
	if (cosi < 0) { cosi = -cosi; }
	else {
		//std::swap(etai, etat);
		float tmp = etat;
		etat = etai;
		etai = tmp;
		n = -N;
	}
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? jek::Vec3f(0.f) : eta * I + (eta * cosi - sqrt(k)) * n;
}
__device__ float power_heuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = (float)nf * fPdf;
	float g = (float)ng * gPdf;
	return (f * f) / (f * f + g * g);
}

__device__ jek::Vec3f fresnel_schlick(const jek::Vec3f& f0, const jek::Vec3f& v, const jek::Vec3f& h) {
	float v_dot_h = fmax(dot(v,h), 0.f);
	return f0 + (jek::Vec3f(1.f, 1.f, 1.f) - f0) * pow(1.f - v_dot_h, 5.f);
}
__device__ jek::Vec3f fresnel_roughness(const jek::Vec3f& f0, const jek::Vec3f& n, const jek::Vec3f& wo, const float r)
{
	return f0 + (max(jek::Vec3f(1.f - r), f0) - f0) * pow(jek::clamp(1.f - fmax(0.f, dot(n, wo)), 0.f, 1.f), 5.f);
}

__device__ float ndf_ggx_tr(const jek::Vec3f& n, const jek::Vec3f& h, const float r)
{
	float a = r * r;
	float a2 = a * a;

	float n_dot_h = fmax(dot(n, h), epsilon);
	float n_dot_h_2 = n_dot_h * n_dot_h;

	float denom = fmax(n_dot_h_2 * (a2 - 1.f) + 1.f, epsilon);

	return a2 / (jek::M_PI * denom * denom);
};
__device__ float ndf_beckmann(const jek::Vec3f& n, const jek::Vec3f& h, const float r)
{
	float a = r * r;
	float a2 = a * a;

	float n_dot_h = fmax(dot(n,h), epsilon);
	float n_dot_h_2 = n_dot_h * n_dot_h;
	float n_dot_h_4 = n_dot_h_2 * n_dot_h_2;

	return 1.f / fmax((jek::M_PI * a2 * n_dot_h_4)* exp((n_dot_h_2 - 1.f) / (a2 * n_dot_h_2)), epsilon);
}

__device__ float g1_ggx(const jek::Vec3f& v, const jek::Vec3f& n, const float r)
{
	float a = r * r;
	float a2 = a * a;
	float n_dot_v = fmax(dot(n, v), epsilon);

	return (n_dot_v * 2.f) / fmax(n_dot_v + sqrt(a2 + (1.f - a2) * n_dot_v * n_dot_v), epsilon);
}
__device__ float g1_beckmann(const jek::Vec3f& v, const jek::Vec3f& n, const float r)
{
	float a = r * r;
	float n_dot_v = fmax(dot(n, v), epsilon);

	float c = n_dot_v / fmax(a * sqrt(1.f - n_dot_v * n_dot_v), epsilon);
	float c2 = c * c;

	float d = 1.f / fmax(a * sqrt(1.f - n_dot_v * n_dot_v), epsilon);

	float g_beckmann = (3.535f * c + 2.181f * c2) / fmax(1.f + 2.276f * c + 2.577f * c2, epsilon);

	return c < 1.6f ? g_beckmann : 1.f;
}
__device__ float g1_schlick_beckmann(const jek::Vec3f& v, const jek::Vec3f& n, const float r)
{
	float a = r * r;
	float k = a * sqrt(2.f / jek::M_PI);

	float n_dot_v = fmax(dot(n, v), epsilon);

	return n_dot_v / fmax(n_dot_v * (1.f - k) + k, epsilon);
}
__device__ float g1_schlick_ggx(const jek::Vec3f& v, const jek::Vec3f& n, const float r)
{
	float a = r * r;
	float k = a / 2.f;

	float n_dot_v = fmax(dot(n, v), epsilon);

	return n_dot_v / fmax(n_dot_v * (1.f - k) + k, epsilon);
}

__device__ float geo_atten_ggx(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r)
{
	return g1_ggx(wi, n, r) * g1_ggx(wo, n, r);
};
__device__ float geo_atten_beckmann(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r)
{
	return g1_beckmann(wi, n, r) * g1_beckmann(wo, n, r);
};
__device__ float geo_atten_schlick_beckmann(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r)
{
	return g1_schlick_beckmann(wi, n, r) * g1_schlick_beckmann(wo, n, r);
};
__device__ float geo_atten_schlick_ggx(const jek::Vec3f& wi, const jek::Vec3f& wo, const jek::Vec3f& n, const float r)
{
	return g1_schlick_ggx(wi, n, r) * g1_schlick_ggx(wo, n, r);
};

__device__ jek::Vec3f diff_get_wi(const Isect& isect)
{
	float e0 = jek::rand_float();
	float e1 = jek::rand_float();

	float sinTheta = sqrt(1.f - e0 * e0);
	float phi = 2.f * jek::M_PI * e1;
	float x = sinTheta * cos(phi);
	float z = sinTheta * sin(phi);
	jek::Vec3f sp = jek::Vec3f(x, e0, z);

	jek::Vec3f N = get_normal(isect);
	jek::Vec3f T = jek::gram_schmidt(N);
	jek::Vec3f B = jek::normalize(jek::cross(N, T));

	jek::Vec3f wi = jek::normalize(T*sp.x + N*sp.y + B*sp.z);

	//if (wi == jek::Vec3f(0)) {
	//	printf("theta: %f\tphi: %f\n", sinTheta, phi);
	//}

	return (wi);
};
__device__ float diff_get_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	return jek::M_1_2PI;
};
__device__ jek::Vec3f diff_get_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	float m = get_metallic(isect);
	jek::Vec3f a = get_albedo(isect);
	jek::Vec3f n = get_normal(isect);

	float n_dot_wi = fmax(jek::dot(n, wi), epsilon);

	jek::Vec3f f0 = jek::mix(isect.material->fresnel, a, m);

	jek::Vec3f wh = normalize(wo + wi);
	jek::Vec3f F = fresnel_schlick(f0, wh, wo);

	jek::Vec3f kD = jek::Vec3f(1) - F;
	kD *= 1.f - m;
	
	return (kD * a * n_dot_wi * jek::M_1_PI);
};

__device__ jek::Vec3f spec_get_wi(const Isect& isect, const jek::Vec3f& wo)
{
	float r = get_roughness(isect);
	float a2 = r * r * r * r;

	float e0 = jek::rand_float();
	float e1 = jek::rand_float();

	float theta = acos(sqrt((1.f - e0) / (e0 * (a2 - 1.f) + 1.f)));
	float phi = jek::M_2PI * e1;
	
	jek::Vec3f h = jek::Vec3f(
		sin(theta) * cos(phi),
		cos(theta),
		sin(theta) * sin(phi)
	);

	jek::Vec3f N = get_normal(isect);
	jek::Vec3f T = jek::gram_schmidt(N);
	jek::Vec3f B = jek::normalize(jek::cross(N, T));

	jek::Vec3f sample = jek::normalize(T * h.x + N * h.y + B * h.z);
	jek::Vec3f wi = jek::normalize(jek::reflect(-wo, sample));

	//if (wi == jek::Vec3f(0)) {
	//	printf("theta: %f\tphi: %f\n", theta, phi);
	//}

	return (wi);
};
__device__ float spec_get_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	float r = get_roughness(isect);
	jek::Vec3f n = get_normal(isect);

	jek::Vec3f wh = jek::normalize(wo + wi);

	float wh_dot_n = fmax(jek::dot(wh, n), epsilon);
	float wo_dot_wh = fmax(jek::dot(wo, wh), epsilon);

	float D = ndf_ggx_tr(n, wh, r);

	return (D * wh_dot_n) / fmax((4.f * wo_dot_wh), epsilon);
};
__device__ jek::Vec3f spec_get_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	float r = get_roughness(isect);
	float m = get_metallic(isect);
	jek::Vec3f a = get_albedo(isect);
	jek::Vec3f f0 = jek::mix(isect.material->fresnel, a, m);

	jek::Vec3f n = get_normal(isect);
	jek::Vec3f wh = jek::normalize(wo + wi);

	float n_dot_wi = fmax(jek::dot(n, wi), epsilon);
	float n_dot_wo = fmax(jek::dot(n, wo), epsilon);
	float wi_dot_wh = fmax(jek::dot(wi, wh), epsilon);

	float D = ndf_ggx_tr(n, wh, r);
	float G = geo_atten_schlick_ggx(wi, wo, n, r);
	jek::Vec3f F = fresnel_schlick(f0, wh, wo);

	jek::Vec3f L = (D*G*F) * n_dot_wi / fmax((4.f * n_dot_wo * n_dot_wi), epsilon);

	return (L);
};

__device__ jek::Vec3f BRDF_L(const dLight* light, const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	jek::Vec3f f, Li, Ld;
	float brdf_pdf = 1.f, light_pdf = 1.f;
	float spec_weight, diff_weight, weight;

	float r = get_roughness(isect);
	jek::Vec3f n = get_normal(isect);
	
	// Sample Light
	light_pdf = light->pdf(isect, wi);
		
	if (light_pdf > 0.f) 
	{
		if (light->is_delta())
		{
			brdf_pdf = (diff_get_pdf(isect, wi, wo) + spec_get_pdf(isect, wi, wo)) * 0.5f;
			weight = power_heuristic(1, light_pdf, 1, brdf_pdf);
		}

		if (weight >= 0.f)
		{
			f = diff_get_f(isect, wi, wo) + spec_get_f(isect, wi, wo);
			Li = light->L(isect, wi);
			Ld += (f * Li / light_pdf) * weight;
		}
	}
	
	// Sample BRDF
	if (!light->is_delta()) 
	{
		jek::Vec3f s_wi = (jek::rand_float() < 0.5) ? spec_get_wi(isect, wo) : diff_get_wi(isect);

		brdf_pdf = (diff_get_pdf(isect, s_wi, wo) + spec_get_pdf(isect, s_wi, wo)) * 0.5f;
		light_pdf = light->pdf(isect, s_wi);

		weight = power_heuristic(1, brdf_pdf, 1, light_pdf);

		if (brdf_pdf > 0.f && weight >= 0.f) {
			f = spec_get_f(isect, s_wi, wo) + diff_get_f(isect, s_wi, wo);
			Li = light->L(isect, s_wi);
			Ld += (f * Li / brdf_pdf) * weight;
		}
	}
	
	return (Ld);
}

__device__ jek::Vec3f BRDF_f(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	return diff_get_f(isect, wi, wo) + diff_get_f(isect, wi, wo);
}
__device__ float BRDF_pdf(const Isect& isect, const jek::Vec3f& wi, const jek::Vec3f& wo)
{
	float spec_pdf = spec_get_pdf(isect, wi, wo);
	float diff_pdf = diff_get_pdf(isect, wi, wo);

	return 0.5 * (spec_pdf + diff_pdf);
}

__device__ jek::Vec3f emissive_L(const Isect& isect, const jek::Vec3f& ray_dir)
{
	if (dot(-get_normal(isect), ray_dir) > 0.f)
		return (isect.material->radiance * isect.material->emissive_color_factor);
	else
		return (jek::Vec3f(0.f));
};
__device__ jek::Vec3f emissive_L(const Isect& isect)
{
	return (isect.material->radiance * isect.material->emissive_color_factor);
};
__device__ jek::Vec3f emissive_L(const dMaterial* material)
{
	return (material->radiance * material->emissive_color_factor);
};