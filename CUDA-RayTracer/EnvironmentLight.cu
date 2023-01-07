#include "EnvironmentLight.h"
#include "dMath.h"
#include "Helpers.cuh"
#include "dTexture.cuh"

#include "Random.h"

#include "light_initialization_kernels.cuh"

__device__ void dEnvironmentLight::compute_dir(const Isect& isect, jek::Vec3f& wi) const
{
	if ((int)hrd_texture == -1 || light_type == Color)
	{
		float u = jek::rand_float();
		float v = jek::rand_float();
		wi = jek::sample_spherical_direction(jek::Vec2f(u, v));
	}
	else 
	{
		float ex = jek::rand_float();
		float ey = jek::rand_float();

		int y = upper_bound(marginal_y, tex_height, ey) - 1.f;

		float* _conds_y = (float*)((char*)conds_y + y * conds_y_pitch);
		int x = upper_bound(_conds_y, tex_width, ex) - 1.f;

		float u = (float)x / (float)tex_width;
		float v = (float)y / (float)tex_height;

		wi = jek::sample_spherical_direction(jek::Vec2f(u, v));
	}
}
__device__ jek::Vec3f dEnvironmentLight::compute_L(const Isect& isect, const jek::Vec3f& wi) const
{
	if ((int)hrd_texture == -1 || light_type == Color)
	{
		return color * ls;
	}
	else 
	{
		jek::Vec2f uv = jek::sample_spherical_map(wi);

		jek::Vec3f s = tex2DLod<float4>(hrd_texture, uv.x, uv.y, 0);
		return s;
	}
}
__device__ jek::Vec3f dEnvironmentLight::compute_L(const Isect& isect) const
{
	if ((int)hrd_texture == -1 || light_type == Color)
	{
		return color * ls;
	}
	else 
	{
		jek::Vec3f wi;
		dir(isect, wi);

		jek::Vec2f uv = jek::sample_spherical_map(wi);

		jek::Vec3f s = tex2DLod<float4>(hrd_texture, uv.x, uv.y, 0);
		return s;
	}
}
__device__ float dEnvironmentLight::compute_pdf(const Isect& isect, const jek::Vec3f& wi) const
{
	if ((int)hrd_texture == -1 || light_type == Color)
	{
		return jek::M_1_4PI;
	}
	else 
	{
		jek::Vec2f uv = jek::sample_spherical_map(wi);

		float pdf;
		surf2Dread(&pdf, pdf_texture, (int)(uv.x * (tex_width-1)) * sizeof(float), (int)(uv.y * (tex_height-1)));

		float sin_theta = sin(jek::M_PI * uv.y);

		if (sin_theta == 0.f) return 0.f;
		else {
			return pdf * (tex_width * tex_height) / (2.f * sin_theta * jek::M_PI * jek::M_PI);
		}
	}
}
__device__ float dEnvironmentLight::compute_pdf(const Isect& isect) const
{
	if ((int)hrd_texture == -1 || light_type == Color)
	{
		return jek::M_1_4PI;
	}
	else 
	{
		jek::Vec3f wi;
		dir(isect, wi);

		jek::Vec2f uv = jek::sample_spherical_map(wi);

		float pdf;
		surf2Dread(&pdf, pdf_texture, (int)(uv.x * (tex_width-1)) * sizeof(float), (int)(uv.y * (tex_height-1)));

		float sin_theta = sin(jek::M_PI * uv.y);

		if (sin_theta == 0.f) return 0.f;
		else {
			return pdf * (tex_width * tex_height) / (2.f * sin_theta * jek::M_PI * jek::M_PI);
		}
	}
}

void EnvironmentLight::init_buffers()
{
	// skybox VAO
	glGenVertexArrays(1, &skyboxVAO);
	glGenBuffers(1, &skyboxVBO);
	glBindVertexArray(skyboxVAO);
	glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVerts), &skyboxVerts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glGenVertexArrays(1, &screenVAO);
	glGenBuffers(1, &screenVBO);
	glBindVertexArray(screenVAO);
	glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(screenVerts), &screenVerts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
}
void EnvironmentLight::init_fbo()
{
	fbo = std::make_shared<FrameBuffer>(size, size);
	fbo->attach(GL_COLOR, GL_RG16F, GL_RG, GL_FLOAT); // used for brdfLUT
	fbo->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
	fbo->attach_rbo(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24);
	fbo->construct();
}
void EnvironmentLight::init_dptr()
{
	checkCudaErrors(cudaMallocManaged(&d_light, sizeof(dEnvironmentLight)));

	auto _d_light = static_cast<dEnvironmentLight*>(d_light);
	init_light_on_device(_d_light);

	auto tex_width = get_tex_width();
	auto tex_height = get_tex_height();

	_d_light->hrd_texture = (hdri_enviromentMap != nullptr) ? load_texture_float(hdri_enviromentMap->filepath) : -1;
	_d_light->tex_width = tex_width;
	_d_light->tex_height = tex_height;
	_d_light->pdf_texture = (hdri_enviromentMap != nullptr) ? create_surface_float(tex_width, tex_height, 1) : -1;
	_d_light->light_type = Color;
	_d_light->color = color;
	_d_light->ls = ls;

	cudaMallocPitch(&_d_light->conds_y, &_d_light->conds_y_pitch, tex_width * sizeof(double), tex_height);
	cudaMalloc(&_d_light->marginal_y, tex_height * sizeof(double));
	cudaMalloc(&_d_light->marginal_p, tex_height * sizeof(double));

	build_environment_light(_d_light);
}
void EnvironmentLight::build_environmentMap_color()
{
	basicBackgroundShader.use();
	basicBackgroundShader.setMat4("projection", captureProjection);
	basicBackgroundShader.setVec3("color", color);

	fbo->bind(size, size);
	for (unsigned int i = 0; i < 6; ++i)
	{
		basicBackgroundShader.setMat4("view", captureViews[i]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, environmentMap->getID(), 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		draw_skybox();
	}
	fbo->unbind();

	glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}
void EnvironmentLight::build_environmentMap_texture()
{
	equirectangularToCubemapShader.use();
	equirectangularToCubemapShader.setInt("equirectangularMap", 0);
	equirectangularToCubemapShader.setMat4("projection", captureProjection);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, hdri_enviromentMap->getID());

	fbo->bind(size, size);
	for (unsigned int i = 0; i < 6; ++i)
	{
		equirectangularToCubemapShader.setMat4("view", captureViews[i]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, environmentMap->getID(), 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		draw_skybox();
	}
	fbo->unbind();

	glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}
void EnvironmentLight::build_irradianceMap()
{
	irradianceShader.use();
	irradianceShader.setInt("equirectangularMap", 0);
	irradianceShader.setMat4("projection", captureProjection);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());

	fbo->bind(32, 32);
	for (unsigned int i = 0; i < 6; ++i)
	{
		irradianceShader.setMat4("view", captureViews[i]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap->getID(), 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		draw_skybox();
	}
	fbo->unbind();
}
void EnvironmentLight::build_prefilterMap()
{
	prefilterShader.use();
	prefilterShader.setInt("environmentMap", 0);
	prefilterShader.setMat4("projection", captureProjection);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());

	fbo->bind(size, size);
	unsigned int maxMipLevels = 5;
	for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
	{
		// reisze framebuffer according to mip-level size.
		unsigned int mipWidth = 128 * std::pow(0.5, mip);
		unsigned int mipHeight = 128 * std::pow(0.5, mip);

		fbo->bind_rbo(mipWidth, mipHeight);
		float roughness = (float)mip / (float)(maxMipLevels - 1);
		prefilterShader.setFloat("roughness", roughness);
		for (unsigned int i = 0; i < 6; ++i)
		{
			prefilterShader.setMat4("view", captureViews[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap->getID(), mip);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			draw_skybox();
		}
	}
	fbo->unbind_rbo();
	fbo->unbind();
}
void EnvironmentLight::init_brdfLUT()
{
	fbo->bind(size, size);
	fbo->bind_rbo(size, size);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo->get_color_tex_id(), 0);
	brdfShader.use();
	draw_screen();
	fbo->unbind_rbo();
	fbo->unbind();
}
void EnvironmentLight::draw_skybox()
{
	glBindVertexArray(skyboxVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
}
void EnvironmentLight::draw_screen()
{
	glBindVertexArray(screenVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

EnvironmentLight::EnvironmentLight() :
	backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
	irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
	prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
	equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
	brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
	basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
	atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
{
	size = 4;

	init_buffers();
	init_fbo();

	environmentMap = std::make_shared<CubeMap>(4, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_environmentMap_color();

	irradianceMap = environmentMap;
	prefilterMap = environmentMap;

	init_brdfLUT();

	init_dptr();
}
EnvironmentLight::EnvironmentLight(glm::vec3 _color) :
	backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
	irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
	prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
	equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
	brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
	basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
	atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
{
	id = gen_id();

	size = 4;
	color = _color;

	init_buffers();
	init_fbo();

	environmentMap = std::make_shared<CubeMap>(4, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_environmentMap_color();

	irradianceMap = environmentMap;
	prefilterMap = environmentMap;

	init_brdfLUT();

	init_dptr();
}
EnvironmentLight::EnvironmentLight(string const& path) :
	backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
	irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
	prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
	equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
	brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
	basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
	atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
{
	id = gen_id();

	light_type = HRDI;

	size = 512;
	hdri_enviromentMap = std::make_shared<Texture>(path, GL_TEXTURE_2D, true);

	init_buffers();
	init_fbo();

	environmentMap = std::make_shared<CubeMap>(512, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_environmentMap_texture();

	irradianceMap = std::make_shared<CubeMap>(32, false, false, "irradiance_map", GL_CLAMP_TO_EDGE);
	build_irradianceMap();

	prefilterMap = std::make_shared<CubeMap>(128, true, false, "pre_filter_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_prefilterMap();

	init_brdfLUT();

	init_dptr();
}
EnvironmentLight::~EnvironmentLight()
{
	if (hdri_enviromentMap != nullptr) glDeleteTextures(1, &environmentMap->id);
	if (hdri_enviromentMap != nullptr) glDeleteTextures(1, &irradianceMap->id);
	if (hdri_enviromentMap != nullptr) glDeleteTextures(1, &prefilterMap->id);
	if (hdri_enviromentMap != nullptr) glDeleteTextures(1, &hdri_enviromentMap->id);
	if (hdri_enviromentMap != nullptr) glDeleteTextures(1, &hdri_enviromentMap->id);
}

GLuint EnvironmentLight::getCubeMapID()
{
	return environmentMap->getID();
}
GLuint EnvironmentLight::getIrradianceMapID()
{
	return irradianceMap->getID();
}
void EnvironmentLight::set_type(EnvironmentLightType type)
{
	light_type = type;
	auto _d_light = static_cast<dEnvironmentLight*>(d_light);
	_d_light->light_type = type;
	notify("");
}
GLuint EnvironmentLight::getPrefilterMapID()
{
	return prefilterMap->getID();
}
GLuint EnvironmentLight::get_brdfLUT_ID()
{
	return fbo->get_color_tex_id();
}
void EnvironmentLight::draw_background(std::shared_ptr<Camera> camera)
{
	glDepthFunc(GL_LEQUAL);

	backgroundShader.use();
	camera->send_uniforms2(backgroundShader);

	// skybox cube
	glBindVertexArray(skyboxVAO);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
}
std::string EnvironmentLight::get_texture_filepath() 
{
	if (hdri_enviromentMap == nullptr) return "NULL";
	else return hdri_enviromentMap->filepath;
}
void EnvironmentLight::set_texture_filepath(std::string filepath)
{
	size = 512;
	hdri_enviromentMap = std::make_shared<Texture>(filepath, GL_TEXTURE_2D, true);

	environmentMap = std::make_shared<CubeMap>(512, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_environmentMap_texture();

	irradianceMap = std::make_shared<CubeMap>(32, false, false, "irradiance_map", GL_CLAMP_TO_EDGE);
	build_irradianceMap();

	prefilterMap = std::make_shared<CubeMap>(128, true, false, "pre_filter_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_prefilterMap();

	init_brdfLUT();

	auto _d_light = static_cast<dEnvironmentLight*>(d_light);

	cudaFree(_d_light->conds_y);
	cudaFree(_d_light->marginal_y);
	cudaFree(_d_light->marginal_p);

	destroy_texture(_d_light->hrd_texture);
	destroy_surface(_d_light->pdf_texture);

	auto tex_width = get_tex_width();
	auto tex_height = get_tex_height();

	_d_light->hrd_texture = (hdri_enviromentMap != nullptr) ? load_texture_float(hdri_enviromentMap->filepath) : -1;
	_d_light->tex_width = tex_width;
	_d_light->tex_height = tex_height;
	_d_light->pdf_texture = (hdri_enviromentMap != nullptr) ? create_surface_float(tex_width, tex_height, 1) : -1;

	cudaMallocPitch(&_d_light->conds_y, &_d_light->conds_y_pitch, tex_width * sizeof(double), tex_height);
	cudaMalloc(&_d_light->marginal_y, tex_height * sizeof(double));
	cudaMalloc(&_d_light->marginal_p, tex_height * sizeof(double));

	build_environment_light(_d_light);
	notify("");
}
glm::vec3 EnvironmentLight::get_color()
{
	return color;
}
void EnvironmentLight::set_color(glm::vec3 color)
{
	(this)->color = color;

	size = 4;

	environmentMap = std::make_shared<CubeMap>(size, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	build_environmentMap_color();

	irradianceMap = environmentMap;
	prefilterMap = environmentMap;

	init_brdfLUT();

	d_light->color = color;
	notify("");
}
dEnvironmentLight* EnvironmentLight::get_dptr() const
{
	auto _d_light = static_cast<dEnvironmentLight*>(d_light);
	return _d_light;
}

int EnvironmentLight::get_tex_width()
{
	if (light_type == Color) {
		return 1;
	}
	else if (light_type == HRDI) {
		if (hdri_enviromentMap == nullptr) return 1;
		else return hdri_enviromentMap->width;
	}
}
int EnvironmentLight::get_tex_height()
{
	if (light_type == Color) {
		return 1;
	}
	else if (light_type == HRDI) {
		if (hdri_enviromentMap == nullptr) return 1;
		else return hdri_enviromentMap->height;
	}
}
GLuint EnvironmentLight::get_hrdi_tex()
{
	return (hdri_enviromentMap == nullptr) ? -1 : hdri_enviromentMap->id;
}
EnvironmentLightType EnvironmentLight::get_light_type()
{
	return light_type;
};
