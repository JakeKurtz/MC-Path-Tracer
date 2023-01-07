#include "Overlay.h"

Overlay::Overlay(int SCR_WIDTH, int SCR_HEIGHT) : 
    shd_solid_color("../shaders/curves/solid_color_vs.glsl", "../shaders/curves/solid_color_fs.glsl"),
    shd_black("../shaders/curves/black_vs.glsl", "../shaders/curves/black_fs.glsl"),
    shd_grid_overlay("../shaders/curves/grid_overlay_vs.glsl", "../shaders/curves/grid_overlay_fs.glsl"),
    shd_draw_texture("../shaders/curves/draw_texture_vs.glsl", "../shaders/curves/draw_texture_fs.glsl")
{
    (this)->SCR_WIDTH = SCR_WIDTH;
    (this)->SCR_HEIGHT = SCR_HEIGHT;

    foo = glm::ortho(-100.f, 100.f, -100.f, 100.f, 0.1f, 10000.f);
    bar = glm::lookAt(glm::vec3(0.f,0.f,10.f), glm::vec3(0.f,-1.f,10.f), glm::vec3(0,0,1));

    init_fbos();
}

void Overlay::draw_wireframe(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    fbo_firstpass->bind(film_width, film_height);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glLineWidth(wireframe.thickness);

    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(0, -1);

    glEnable(GL_DEPTH_TEST);

    shd_solid_color.use();
    for (auto const& r_obj : scene->render_objects) {

        auto object = r_obj.second;
        auto obj_id = r_obj.second;

        if (object != nullptr) {
            shd_solid_color.setVec4("color", wireframe.color);
            shd_solid_color.setMat4("model", object->get_transform()->get_matrix());
            camera->send_uniforms(shd_solid_color);
            object->draw(shd_solid_color);
        }
    }
    glDisable(GL_POLYGON_OFFSET_LINE);

    fbo_firstpass->unbind();

    glDisable(GL_DEPTH_TEST);

    out = fbo_firstpass->get_color_tex_id();
}

void Overlay::depth_pass(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    fbo_firstpass->bind(film_width, film_height);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    shd_black.use();
    for (auto const& r_obj : scene->render_objects) {

        auto object = r_obj.second;
        auto obj_id = r_obj.second;

        if (object != nullptr) {
            shd_black.setMat4("model", object->get_transform()->get_matrix());
            camera->send_uniforms(shd_black);
            object->draw(shd_black);
        }
    }

    fbo_firstpass->unbind();

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
}

GLuint Overlay::draw(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    GLuint overlay;

    depth_pass(scene, camera, film);

    if (enable_wireframe) draw_wireframe(overlay, scene, camera, film);
    if (enable_grid) {
        if (grid_flat) {
            draw_flat_grid(overlay, scene, camera, film);
        }
        else {
            draw_perspective_grid(overlay, scene, camera, film);
        }
    }

    fbo_main->bind(film_width, film_height);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    shd_draw_texture.use();

    shd_draw_texture.setVec2("tex_dim", glm::vec2(film_width, film_height));
    shd_draw_texture.setVec2("scr_dim", glm::vec2(film_width, film_height));

    shd_draw_texture.setInt("tex", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, film->get_image_tex());
    render_screen();

    shd_draw_texture.setVec2("tex_dim", glm::vec2(film_width, film_height));
    shd_draw_texture.setVec2("scr_dim", glm::vec2(SCR_WIDTH, SCR_HEIGHT));

    shd_draw_texture.setInt("tex", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, overlay);
    render_screen();

    fbo_main->unbind();

    glDisable(GL_BLEND);

    film->copy_texture_to_film(fbo_main->get_color_tex_id());

    return fbo_main->get_color_tex_id();
}

void Overlay::draw_flat_grid(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    fbo_secondpass->bind(film_width, film_height);

    GLuint depth_tex = fbo_firstpass->get_depth_tex_id();
    GLuint wireframe_tex = fbo_firstpass->get_color_tex_id();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    shd_grid_overlay.use();

    camera->send_uniforms(shd_grid_overlay);

    shd_grid_overlay.setVec2("tex_dim", glm::vec2(film_width, film_height));
    shd_grid_overlay.setVec2("scr_dim", glm::vec2(SCR_WIDTH, SCR_HEIGHT));

    shd_grid_overlay.setFloat("grid_scale", grid.scale);
    shd_grid_overlay.setFloat("max_dist_line_width", grid.max_dist_line_width);
    shd_grid_overlay.setFloat("min_dist_line_width", grid.min_dist_line_width);
    shd_grid_overlay.setFloat("near", grid.fade_near);
    shd_grid_overlay.setFloat("far", grid.fade_far);
    shd_grid_overlay.setInt("fade_mode", grid.fade_mode);
    shd_grid_overlay.setBool("highlight_axis", grid.highlight_axis);
    shd_grid_overlay.setBool("perspective_mode", true);
    shd_grid_overlay.setVec4("color", grid.color);

    shd_grid_overlay.setInt("depth_tex", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depth_tex);

    shd_grid_overlay.setInt("wireframe_tex", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, wireframe_tex);

    shd_grid_overlay.setMat4("model", glm::mat4(1.f));
    shd_grid_overlay.setMat4("view", bar);
    shd_grid_overlay.setMat4("projection", foo);

    render_screen();

    fbo_secondpass->unbind();

    glDisable(GL_DEPTH_TEST);

    out = fbo_secondpass->get_color_tex_id();
}

void Overlay::draw_perspective_grid(GLuint& out, std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    fbo_secondpass->bind(film_width, film_height);

    GLuint depth_tex = fbo_firstpass->get_depth_tex_id();
    GLuint wireframe_tex = fbo_firstpass->get_color_tex_id();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    shd_grid_overlay.use();

    shd_grid_overlay.setVec2("tex_dim", glm::vec2(film_width, film_height));
    shd_grid_overlay.setVec2("scr_dim", glm::vec2(SCR_WIDTH, SCR_HEIGHT));

    shd_grid_overlay.setFloat("grid_scale", grid.scale);
    shd_grid_overlay.setFloat("max_dist_line_width", grid.max_dist_line_width);
    shd_grid_overlay.setFloat("min_dist_line_width", grid.min_dist_line_width);
    shd_grid_overlay.setFloat("near", grid.fade_near);
    shd_grid_overlay.setFloat("far", grid.fade_far);
    shd_grid_overlay.setInt("fade_mode", grid.fade_mode);
    shd_grid_overlay.setBool("highlight_axis", grid.highlight_axis);
    shd_grid_overlay.setBool("perspective_mode", false);
    shd_grid_overlay.setVec4("color", grid.color);

    shd_grid_overlay.setInt("depth_tex", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depth_tex);

    shd_grid_overlay.setInt("wireframe_tex", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, wireframe_tex);

    // recentering, so the grid will be rendered at the origin.
    glm::mat4 matrix_recenter = glm::translate(grid.transform->get_matrix(), grid.transform->get_centroid());

    shd_grid_overlay.setMat4("model", matrix_recenter);
    camera->send_uniforms(shd_grid_overlay);
    render_screen();

    fbo_secondpass->unbind();

    glDisable(GL_DEPTH_TEST);

    out = fbo_secondpass->get_color_tex_id();
}

void Overlay::render_screen()
{
    if (screenVAO == 0)
    {
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
    glBindVertexArray(screenVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Overlay::init_fbos()
{
    fbo_main = make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo_main->attach(GL_COLOR, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE); // color
    fbo_main->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_main->construct();

    fbo_firstpass = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo_firstpass->attach(GL_COLOR, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE); // color
    fbo_firstpass->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_firstpass->construct();

    fbo_secondpass = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo_secondpass->attach(GL_COLOR, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE); // color
    fbo_secondpass->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_secondpass->construct();
}
