#include "MaterialPreviewWindow.h"

MaterialPreviewWindow::MaterialPreviewWindow(int width, int height, std::shared_ptr<RenderEngine> render_engine) : RenderWindow(render_engine)
{
    this->name = "Material View##" + to_string(gen_id());

    //this->path_tracer = make_shared<PathTracer>(width, height);

    transform_edit = make_shared<Transform>();

    init_scene();
}

void MaterialPreviewWindow::init_scene()
{
    film = std::make_shared<Film>();

    std::shared_ptr<DirectionalLight> light = std::make_shared<DirectionalLight>();
    light->set_ls(10.f);
    light->set_dir(glm::vec3(10.f));
    light->set_color(glm::vec3(1.f));

    env_light_hrd = make_shared<EnvironmentLight>("../hrdi/night_free_Env.hdr");

    this->scene = make_shared<Scene>();

    scene->add_light(light);
    scene->set_environment_light(env_light_hrd);

    scene->load("../models/mat_preview.glb");

    film = std::make_shared<Film>();
    camera = std::make_shared<PerspectiveCamera>(glm::vec3(0.2f, 3.4f, 4.f), glm::radians(45.f), 0.01f, 10000.f);
    camera->set_yaw_pitch(-90.f, -30.3000793);
    camera->attach(film);
    scene->attach(film);
}

void MaterialPreviewWindow::draw_windowless()
{
    render_engine->render(scene, camera, film, RenderMode::mode_rasterizer, 0);
}

void MaterialPreviewWindow::render()
{
    render_engine->render(scene, camera, film, RenderMode::mode_rasterizer, 0);
}

shared_ptr<Material> MaterialPreviewWindow::get_material()
{
    return scene->materials_loaded["mat_preview"];
}

void MaterialPreviewWindow::set_material(shared_ptr<Material> material)
{
    scene->render_objects[26]->set_material(material);
}

void MaterialPreviewWindow::window_size_callback(GLFWwindow* window, int width, int height)
{
}

void MaterialPreviewWindow::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
}

void MaterialPreviewWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
}

void MaterialPreviewWindow::mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
}

void MaterialPreviewWindow::key_callback()
{
}

void MaterialPreviewWindow::process_input(GLFWwindow* window)
{
}
