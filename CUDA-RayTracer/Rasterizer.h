#pragma once

#include <cmath>

#include "Light.h"
#include "Shader.h"
#include "FrameBuffer.h"
#include "Scene.h"
#include "EnvironmentLight.h"
#include "TextureArray.h"
#include "Overlay.h"
//#include "Film.cuh"

const int SHADED = 0;
const int POSITION = 1;
const int NORMAL = 2;
const int ALBEDO = 3;
const int MRO = 4;
const int EMISSION = 5;
const int DEPTH = 6;
const int SHADOW = 7;

class Rasterizer 
{
public:
    std::shared_ptr<TextureArray> cascadeShadowMapTexArray;
    std::shared_ptr<TextureArray> cascadeShadowMapTexArray_tmptarget;

    unsigned int depthMap;
    unsigned int depthMapFBO;

    std::shared_ptr<FrameBuffer> fbo;

    std::shared_ptr<FrameBuffer> fbo_sdwmap_tmptarget;
    std::shared_ptr<FrameBuffer> fbo_hdr_tmptarget;
    std::shared_ptr<FrameBuffer> fbo_sdwmap;

    std::shared_ptr<FrameBuffer> fbo_test;
    std::shared_ptr<FrameBuffer> fbo_test2;

    std::shared_ptr<FrameBuffer> gBuffer;

    Shader shaderGeometryPass;
    Shader shaderLightingPass;
    Shader shaderShadowPass;
    Shader shaderShadowDepth;
    Shader gaussFilter;
    Shader bilateralFilter;
    Shader shaderCurve;
    Shader shd_solid_color;
    Shader shd_draw_texture;

    unsigned int uboExampleBlock;

    int shadow_cube_size = 1024;
    int shadow_cascade_size = 4096;

    float cascade_shadow_offset = 250.f;
    float lookat_point_offset = 1.f;
    float lightSize = 250.f;
    float searchAreaSize = 15.f;
    float shadowScaler = 0.05f;
    int kernel_size = 9;

    bool enable_wireframe = true;
    bool enable_xray = false;
    bool enable_grid = true;

    float scale = 10.f;
    glm::mat4 cascade_proj_mat = glm::ortho(-100.f, 100.f, -100.f, 100.f, 0.1f, 10000.f);
    //glm::mat4 cascade_proj_mat = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 10000.f);
    glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);

    unsigned int screenVAO = 0;
    unsigned int screenVBO;
    float screenVerts[20] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    int SCR_WIDTH = 1000;
    int SCR_HEIGHT = 1000;

    Rasterizer(int _SCR_WIDTH = 1000, int _SCR_HEIGHT = 1000);

    void draw_clay(std::shared_ptr<Scene> scene);

    GLuint draw_scene(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film, const int raster_output_type);
    void drawCurves(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    void drawGeometry(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera);
    void drawLighting(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    void drawShadows(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    void drawShadowMaps(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);

    GLuint draw_texture(GLuint tex);

    void setScreenSize(int _SCR_WIDTH, int _SCR_HEIGHT);
    void setShadowCubeSize(int _size);
    void setShadowCascadeSize(int _size);

private:
    void initFBOs();
    void applyFilter(std::shared_ptr<Scene> scene);
    //void applyToneMapping(std::shared_ptr<FrameBuffer> targetFB, aType targetAttachment, unsigned int texId);
    void draw_screen();
};