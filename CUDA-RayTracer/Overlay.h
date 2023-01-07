#pragma once

#include "Scene.h"

struct Raster_Grid {
    int fade_mode = 0;
    float fade_near = 0.001f;
    float fade_far = 20.0f;

    bool highlight_axis = true;

    float scale = 1.f;
    float max_dist_line_width = 0.f;
    float min_dist_line_width = 1.f;

    glm::vec4 color = glm::vec4(0.f, 0.f, 0.f, 1.f);
    std::shared_ptr<Transform> transform =  std::make_shared<Transform>();
};

struct Raster_Wireframe {
    float thickness = 1.f;
    glm::vec4 color = glm::vec4(0.f, 0.f, 0.f, 1.f);
};

class Overlay
{
public:
    Overlay(int _SCR_WIDTH = 1000, int _SCR_HEIGHT = 1000);

    std::shared_ptr<FrameBuffer> fbo_main = nullptr;
    std::shared_ptr<FrameBuffer> fbo_firstpass = nullptr;
    std::shared_ptr<FrameBuffer> fbo_secondpass = nullptr;

    Shader shd_solid_color;
    Shader shd_draw_texture;
    Shader shd_black;
    Shader shd_grid_overlay;

    Raster_Grid grid;
    Raster_Wireframe wireframe;

    glm::mat4 foo;
    glm::mat4 bar;

    bool enable_wireframe = true;
    bool enable_xray = false;
    bool enable_grid = true;
    bool grid_flat = false;

    unsigned int screenVAO = 0;
    unsigned int screenVBO;
    float screenVerts[20] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    int SCR_WIDTH;
    int SCR_HEIGHT;

    void depth_pass(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);

    void draw_wireframe(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);

    GLuint draw(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    void draw_flat_grid(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    void draw_perspective_grid(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);

    void render_screen();

    void setScreenSize(int _SCR_WIDTH, int _SCR_HEIGHT)
    {
        //SCR_WIDTH = _SCR_WIDTH;
        //SCR_HEIGHT = _SCR_HEIGHT;

        fbo_main->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
        fbo_firstpass->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
        fbo_secondpass->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
    }

    void init_fbos();
};

