#pragma once
#include "GLCommon.h"
#include "globals.h"

#include "Camera.h"
#include "CubeMap.h"
#include "FrameBuffer.h"

#include "Light.h"

enum EnvironmentLightType {
    Color = 0,
    HRDI = 1,
    Atmosphere = 2
};

struct dEnvironmentLight : public dLight
{
    EnvironmentLightType light_type = Color;

    cudaTextureObject_t hrd_texture = -1;
    cudaSurfaceObject_t pdf_texture = -1;

    float* marginal_y;
    float* marginal_p;
    float* conds_y;
    size_t conds_y_pitch;

    float radius;
    unsigned int tex_width = 1;
    unsigned int tex_height = 1;
    float pdf_denom;

private:
    __device__ void compute_dir(const Isect& isect, jek::Vec3f& wi) const;
    __device__ jek::Vec3f compute_L(const Isect& isect, const jek::Vec3f& wi) const;
    __device__ jek::Vec3f compute_L(const Isect& isect) const;
    __device__ float compute_pdf(const Isect& isect, const jek::Vec3f& wi) const;
    __device__ float compute_pdf(const Isect& isect) const;
};

class EnvironmentLight : public Light
{
private:
    unsigned int skyboxVAO, skyboxVBO;
    float skyboxVerts[108] =
    {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };

    unsigned int screenVAO, screenVBO;
    float screenVerts[20] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    glm::mat4 captureViews[6] =
    {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    glm::vec3 captureDir[6] =
    {
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(-1.0f, 0.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f, -1.0f,  0.0f),
        glm::vec3(0.0f,  0.0f, 1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f)
    };

    glm::vec3 captureUp[6] =
    {
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  0.0f,  1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f)
    };

    glm::vec3 captureRight[6] =
    {
        glm::vec3(0.0f,  0.0f,  1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f),
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(-1.0f,  0.0f,  0.0f),
        glm::vec3(1.0f,  0.0f,  0.0f)
    };

    float* conds_y, *marginal_y, *marginal_p, *pdf_denom;
    cudaSurfaceObject_t pdf_texture;

    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

    std::shared_ptr<FrameBuffer> fbo;

    std::shared_ptr<Texture> hdri_enviromentMap = nullptr;
    glm::vec3 color = glm::vec3(0.8);

    std::shared_ptr<CubeMap> environmentMap;
    std::shared_ptr<CubeMap> irradianceMap;
    std::shared_ptr<CubeMap> prefilterMap;

    Shader atmosphereShader;
    Shader basicBackgroundShader;
    Shader equirectangularToCubemapShader;
    Shader backgroundShader;
    Shader irradianceShader;
    Shader prefilterShader;
    Shader brdfShader;

    EnvironmentLightType light_type = Color;

    unsigned int size;
    unsigned int id;

    void init_buffers();
    void init_fbo();

    void init_dptr();

    void build_environmentMap_color();
    void build_environmentMap_texture();
    void build_irradianceMap();
    void build_prefilterMap();
    void init_brdfLUT();
    void draw_skybox();
    void draw_screen();

    dEnvironmentLight* get_dptr() const;

public:
    EnvironmentLight();
    EnvironmentLight(glm::vec3 _color);
    EnvironmentLight(string const& path);
    ~EnvironmentLight();

    void set_color(glm::vec3 color);
    void set_type(EnvironmentLightType type);
    void set_texture_filepath(std::string filepath);

    GLuint getCubeMapID();
    GLuint getIrradianceMapID();
    GLuint getPrefilterMapID();
    GLuint get_brdfLUT_ID();
    GLuint get_hrdi_tex();

    std::string get_texture_filepath();
    int get_tex_width();
    int get_tex_height();
    glm::vec3 get_color();
    EnvironmentLightType get_light_type();

    void draw_background(std::shared_ptr<Camera> camera);
};

