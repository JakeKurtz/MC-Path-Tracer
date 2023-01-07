#include "Rasterizer.h"

Rasterizer::Rasterizer(int _SCR_WIDTH, int _SCR_HEIGHT) :
    shaderGeometryPass("../shaders/main/geometryPass_vs.glsl", "../shaders/main/geometryPass_fs.glsl"),
    shaderLightingPass("../shaders/main/lightingPass_vs.glsl", "../shaders/main/lightingPass_fs.glsl"),
    shaderShadowPass("../shaders/main/shadowPass_vs.glsl", "../shaders/main/shadowPass_fs.glsl"),
    shaderShadowDepth("../shaders/main/shadowDepth_vs.glsl", "../shaders/main/shadowDepth_fs.glsl"),
    gaussFilter("../shaders/main/gaussFilter_vs.glsl", "../shaders/main/gaussFilter_fs.glsl"),
    bilateralFilter("../shaders/main/bilateralFilter_vs.glsl", "../shaders/main/bilateralFilter_fs.glsl"),
    shaderCurve("../shaders/curves/curve_vs.glsl", "../shaders/curves/curve_fs.glsl"),
    shd_solid_color("../shaders/curves/solid_color_vs.glsl", "../shaders/curves/solid_color_fs.glsl"),
    shd_draw_texture("../shaders/curves/draw_texture_vs.glsl", "../shaders/curves/draw_texture_fs.glsl")
{
    SCR_WIDTH = _SCR_WIDTH;
    SCR_HEIGHT = _SCR_HEIGHT;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_3D);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    //blueNoise = new Texture("LDR_RG01_0.png", "../textures/", "noise", GL_TEXTURE_2D);
    cascadeShadowMapTexArray = std::make_shared<TextureArray>(shadow_cascade_size, shadow_cascade_size, 4, false);
    cascadeShadowMapTexArray_tmptarget = std::make_shared<TextureArray>(shadow_cascade_size, shadow_cascade_size, 4, false);

    uboExampleBlock;
    glGenBuffers(1, &uboExampleBlock);
    glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
    glBufferData(GL_UNIFORM_BUFFER, 32 * sizeof(glm::mat4), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboExampleBlock, 0, 32 * sizeof(glm::mat4));

    unsigned int uniformBlockIndexRed = glGetUniformBlockIndex(shaderLightingPass.ID, "ExampleBlock");

    glUniformBlockBinding(shaderLightingPass.ID, uniformBlockIndexRed, 0);

    initFBOs();
}

GLuint Rasterizer::draw_scene(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film, const int raster_output_type)
{

    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    // ------ GEOMETRY PASS ------ //
    gBuffer->bind(film_width, film_height);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawGeometry(scene, camera);

    gBuffer->unbind();
    
    // ------ SHADOW MAP PASS ------ //
    /*
    glCullFace(GL_FRONT);
    drawShadowMaps(scene);
    glCullFace(GL_BACK);

    fbo_test2->bind();
    drawShadows(scene);
    fbo_test2->unbind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    */
    if (raster_output_type == SHADED) {
        
        // ------ LIGHTING PASS ------ //

        glDisable(GL_DEPTH_TEST);

        fbo->bind(film_width, film_height);

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        drawLighting(scene, camera, film);

        glEnable(GL_DEPTH_TEST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer->get_id());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->get_id());
        glBlitFramebuffer(0, 0, gBuffer->get_width(), gBuffer->get_height(), 0, 0, fbo->get_width(), fbo->get_height(), GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        fbo->unbind();
    }
    else {

        fbo->bind(film_width, film_height);
        switch (raster_output_type) {
            case POSITION:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT0);
                break;
            case NORMAL:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT1);
                break;
            case ALBEDO:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT2);
                break;
            case MRO:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT3);
                break;
            case EMISSION:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT4);
                break;
            case DEPTH:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT5);
                break;
            default:
                glNamedFramebufferReadBuffer(gBuffer->get_id(), GL_COLOR_ATTACHMENT0);
                break;
        }

        glNamedFramebufferDrawBuffer(fbo->get_id(), GL_COLOR_ATTACHMENT0);
        glBlitNamedFramebuffer(gBuffer->get_id(), fbo->get_id(), 0, 0, gBuffer->get_width(), gBuffer->get_height(), 0, 0, gBuffer->get_width(), gBuffer->get_height(), GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        glBlitNamedFramebuffer(gBuffer->get_id(), fbo->get_id(), 0, 0, gBuffer->get_width(), gBuffer->get_height(), 0, 0, gBuffer->get_width(), gBuffer->get_height(), GL_COLOR_BUFFER_BIT, GL_NEAREST);
        fbo->unbind();
    }

    // ------ BACKGROUND ------ //
    fbo->bind(film_width, film_height);
    scene->environment_light->draw_background(camera);
    fbo->unbind();

    film->copy_texture_to_film(fbo->get_color_tex_id());

    return fbo->get_color_tex_id();
}

void Rasterizer::drawGeometry(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera)
{
    shaderGeometryPass.use();
    for (auto const& r_obj : scene->render_objects) {

        auto object = r_obj.second;
        auto obj_id = r_obj.second;

        if (object != nullptr) {
            shaderGeometryPass.setMat4("model", object->get_transform()->get_matrix());
            camera->send_uniforms(shaderGeometryPass);
            object->draw(shaderGeometryPass);
        }
    }
}

void Rasterizer::drawLighting(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);

    shaderLightingPass.use();

    shaderLightingPass.setVec2("tex_dim", glm::vec2(film_width, film_height));
    shaderLightingPass.setVec2("scr_dim", glm::vec2(SCR_WIDTH, SCR_HEIGHT));

    shaderLightingPass.setInt("gPosition", 0);
    shaderLightingPass.setInt("gNormal", 1);
    shaderLightingPass.setInt("gAlbedo", 2);
    shaderLightingPass.setInt("gMetallicRoughAO", 3);
    shaderLightingPass.setInt("gEmissive", 4);
    shaderLightingPass.setInt("gShadows", 5);
    shaderLightingPass.setInt("irradianceMap", 6);
    shaderLightingPass.setInt("prefilterMap", 7);
    shaderLightingPass.setInt("brdfLUT", 8);

    shaderLightingPass.setInt("shadowMaps", 9);

    shaderLightingPass.setFloat("searchAreaSize", searchAreaSize);
    shaderLightingPass.setFloat("lightSize", lightSize);
    shaderLightingPass.setInt("kernel_size", kernel_size);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id(0));

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id(1));

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id(2));

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id(3));

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id(4));

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, fbo_test->get_color_tex_id());

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_CUBE_MAP, scene->environment_light->getIrradianceMapID());

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_CUBE_MAP, scene->environment_light->getPrefilterMapID());

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, scene->environment_light->get_brdfLUT_ID());

    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

    scene->send_uniforms(shaderLightingPass);
    camera->send_uniforms(shaderLightingPass);

    glm::vec3 cam_lookat_pos = camera->get_position();//(camera->front * lookat_point_offset) + camera->position;
    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glm::vec3 dir = -scene->dir_lights[i]->get_dir();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        glm::mat4 LSM = cascade_proj_mat * lookat;

        glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(LSM));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    draw_screen();
}

void Rasterizer::drawShadows(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test2->get_color_tex_id(), 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    shaderShadowPass.use();

    shaderShadowPass.setInt("gPosition", 0);
    shaderShadowPass.setInt("shadowMaps", 1);

    shaderShadowPass.setFloat("searchAreaSize", searchAreaSize);
    shaderShadowPass.setFloat("lightSize", lightSize);
    shaderShadowPass.setFloat("shadowScaler", shadowScaler);
    shaderShadowPass.setInt("kernel_size", kernel_size);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gBuffer->get_color_tex_id());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

    scene->send_uniforms(shaderShadowPass);
    camera->send_uniforms(shaderShadowPass);

    glm::vec3 cam_lookat_pos = camera->get_position();//(camera->front * lookat_point_offset) + camera->position;
    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glm::vec3 dir = -scene->dir_lights[i]->get_dir();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        glm::mat4 LSM = cascade_proj_mat * lookat;

        glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(LSM));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    draw_screen();

    glDisable(GL_DEPTH_TEST);

    // ----- First pass ----- //

    fbo_test->bind();
    bilateralFilter.use();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test->get_color_tex_id(), 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo_test2->get_color_tex_id());

    bilateralFilter.setBool("horizontal", true);
    bilateralFilter.setInt("image", 0);
    bilateralFilter.setVec3("scale", glm::vec3(1.f / (SCR_WIDTH * 0.1f), 0.f, 0.f));
    bilateralFilter.setFloat("r", 16.f);
    draw_screen();

    fbo_test->unbind();
    
    // ----- Second pass ----- //

    fbo_test2->bind();
    gaussFilter.use();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test2->get_color_tex_id(), 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo_test->get_color_tex_id());

    gaussFilter.setBool("horizontal", false);
    gaussFilter.setInt("image", 0);
    gaussFilter.setVec3("scale", glm::vec3(0.f, 1.f / (SCR_HEIGHT * 0.1f), 0.f));
    gaussFilter.setFloat("r", 16.f);
    draw_screen();

    fbo_test2->unbind();

    glEnable(GL_DEPTH_TEST);
    
}

void Rasterizer::drawShadowMaps(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    glm::vec3 cam_lookat_pos = camera->get_position();//(camera->front * lookat_point_offset) + camera->position;

    fbo_sdwmap->bind();
    shaderShadowDepth.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glm::vec3 dir = -scene->dir_lights[i]->get_dir();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        glm::mat4 LSM = cascade_proj_mat * lookat;

        shaderShadowDepth.setMat4("lsm", LSM);
        for (auto const& r_obj : scene->render_objects) {

            auto object = r_obj.second;
            auto obj_id = r_obj.second;

            if (object != nullptr) {
                shaderShadowDepth.setMat4("model", object->get_transform()->get_matrix());
                object->draw(shaderShadowDepth);
            }
        }
    }
    fbo_sdwmap->unbind();

    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
}

GLuint Rasterizer::draw_texture(GLuint tex)
{
    fbo->bind();

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH);

    shd_draw_texture.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    shd_draw_texture.setInt("tex", 0);

    draw_screen();

    glEnable(GL_DEPTH);

    fbo->unbind();

    return fbo->get_color_tex_id();
}

void Rasterizer::setScreenSize(int _SCR_WIDTH, int _SCR_HEIGHT)
{
    fbo->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
    fbo_sdwmap->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
    fbo_test->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
    gBuffer->set_attachment_size(_SCR_WIDTH, _SCR_HEIGHT);
}

void Rasterizer::setShadowCubeSize(int _size)
{
}

void Rasterizer::setShadowCascadeSize(int _size)
{
}

void Rasterizer::initFBOs()
{
    fbo = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo->attach(GL_COLOR, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
    fbo->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo->construct();

    fbo_sdwmap_tmptarget = std::make_shared<FrameBuffer>(shadow_cascade_size, shadow_cascade_size);
    fbo_sdwmap_tmptarget->attach(GL_COLOR, GL_RG32F, GL_RG, GL_FLOAT);
    fbo_sdwmap_tmptarget->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_sdwmap_tmptarget->construct();

    fbo_sdwmap = std::make_shared<FrameBuffer>(shadow_cascade_size, shadow_cascade_size);
    fbo_sdwmap->attach(GL_COLOR, GL_RG32F, GL_RG, GL_FLOAT);
    fbo_sdwmap->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_sdwmap->construct();

    fbo_test = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo_test->attach(GL_COLOR, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    fbo_test->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_test->construct();

    fbo_test2 = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    fbo_test2->attach(GL_COLOR, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    fbo_test2->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_test2->construct();

    // Setup defferred shading
    gBuffer = std::make_shared<FrameBuffer>(SCR_WIDTH, SCR_HEIGHT);
    gBuffer->attach(GL_COLOR, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    gBuffer->attach(GL_COLOR, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    gBuffer->attach(GL_COLOR, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
    gBuffer->attach(GL_COLOR, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    gBuffer->attach(GL_COLOR, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
    gBuffer->attach(GL_DEPTH, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    gBuffer->construct();
}

void Rasterizer::applyFilter(std::shared_ptr<Scene> scene)
{
    glDisable(GL_DEPTH_TEST);

    fbo_sdwmap_tmptarget->bind();
    gaussFilter.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray_tmptarget->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

        gaussFilter.setBool("horizontal", true);
        gaussFilter.setInt("shadowMaps", 0);
        gaussFilter.setInt("index", i);
        gaussFilter.setVec3("scale", glm::vec3(1.f / (2048 * 0.25f), 0.f, 0.f));
        draw_screen();

    }
    fbo_sdwmap_tmptarget->unbind();

    // ----- Second pass ----- //

    fbo_sdwmap->bind();
    gaussFilter.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray_tmptarget->getID());

        gaussFilter.setBool("horizontal", false);
        gaussFilter.setInt("shadowMaps", 0);
        gaussFilter.setInt("index", i);
        gaussFilter.setVec3("scale", glm::vec3(0.f, 1.f / (2048 * 0.25f), 0.f));
        draw_screen();
    }
    fbo_sdwmap->unbind();

    glEnable(GL_DEPTH_TEST);
}

//void Rasterizer::applyToneMapping(std::shared_ptr<FrameBuffer> targetFB, aType targetAttachment, unsigned int texId)
//{
    /*glDisable(GL_DEPTH_TEST);

    targetFB->bind(targetAttachment);
    hdrShader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texId);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    hdrShader.setFloat("exposure", camera->Exposure);
    //hdrShader.setFloat("lum", lum);
    screen->model.Draw(hdrShader);

    glEnable(GL_DEPTH_TEST);
    */
//}

void Rasterizer::draw_screen()
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
