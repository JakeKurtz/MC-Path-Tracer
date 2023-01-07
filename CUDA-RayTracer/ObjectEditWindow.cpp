#include "ObjectEditWindow.h"

int modulo(int x, int m) {
    return (x % m + m) % m;
}

ObjectEditWindow::ObjectEditWindow(std::shared_ptr<RenderEngine> render_engine) : RenderWindow(render_engine)
{
    this->name = "Edit View##" + to_string(gen_id());

    transform_edit = make_shared<Transform>();

    init_scene();
}

void ObjectEditWindow::init_scene()
{
    film = std::make_shared<Film>();

    shared_ptr<DirectionalLight> light = make_shared<DirectionalLight>();
    light->set_ls(10.f);
    light->set_dir(glm::vec3(10.f));
    light->set_color(glm::vec3(1.f));

    this->scene = make_shared<Scene>();

    film = std::make_shared<Film>();
    camera = std::make_shared<PerspectiveCamera>(glm::vec3(0.f, 0.f, 10.f), glm::radians(45.f), 0.01f, 10000.f);
    camera->attach(film);
    scene->attach(film);
}

void ObjectEditWindow::render()
{
    if (r_obj != nullptr)
    {
        r_obj->set_transform(transform_edit);
        render_engine->overlay->grid.transform = transform_edit;

        render_engine->render(scene, camera, film, (RenderMode)render_mode, enable_overlay);

        render_engine->overlay->grid.transform = transform_world;
        r_obj->set_transform(transform_world);
    }
}

void ObjectEditWindow::set_obj(shared_ptr<RenderObject> obj)
{
    if (r_obj != nullptr) {
        scene->remove_render_object(r_obj);
    }

    r_obj = obj;
    scene->add_render_object(obj);

    (this)->transform_world = obj->get_transform();

    centroid = obj->center_of_mass();

    transform_edit->reset();
    transform_edit->set_centroid(centroid);
    transform_edit->translate(-centroid);

    transform_edit->apply();
}

void ObjectEditWindow::window_size_callback(GLFWwindow* window, int width, int height)
{
    printf("window size if changing!!!!");
}

void ObjectEditWindow::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (first_mouse)
    {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }

    if (last_x != xpos || last_y != ypos) {
        if (mouse_button_3_down && !shift_down && focused()) {

            glm::ivec2 main_win_pos;
            glfwGetWindowPos(window, &main_win_pos.x, &main_win_pos.y);

            glm::vec2 foo;
            foo.x = window_pos.x - main_win_pos.x;
            foo.y = window_pos.y - main_win_pos.y;

            glm::vec2 v0 = glm::vec2(last_x, last_y) - foo;
            glm::vec2 v1 = glm::vec2(xpos, ypos) - foo;

            transform_edit->rotate(compute_arcball_quat(v0, v1));
            transform_edit->apply();
        }

        if (mouse_button_3_down && shift_down && focused()) {
            glm::vec2 v0 = glm::vec2(last_x, last_y);
            glm::vec2 v1 = glm::vec2(xpos, ypos);

            glm::vec3 dir = glm::vec3(v0 - v1, 0.f) * 0.01f;

            dir.x = -dir.x;

            transform_edit->translate(dir * camera->get_zoom());
            transform_edit->apply();
        }
    }

    last_x = xpos;
    last_y = ypos;
}

void ObjectEditWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // REFACTOR: Duplicate Code.
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouse_down = true;
        }
        else if (action == GLFW_RELEASE) {
            mouse_down = false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_3) {
        if (action == GLFW_PRESS) {
            mouse_button_3_down = true;
        }
        else if (action == GLFW_RELEASE) {
            mouse_button_3_down = false;
        }
    }
    // END REFACTOR: Duplicate Code.
}

void ObjectEditWindow::mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (focused()) 
    {  
        auto zoom = camera->get_zoom();
        if (yoffset < 0) zoom *= 1.1f;
        if (yoffset > 0) zoom *= 0.9f;
        camera->set_zoom(zoom);
    }
}

void ObjectEditWindow::key_callback()
{
}

void ObjectEditWindow::process_input(GLFWwindow* window)
{
}

glm::vec3 ObjectEditWindow::compute_unit_sphere_point(glm::vec2 p)
{
    float offset = 250;

    p += offset;

    glm::vec3 p_ndc = glm::vec3(0.f);

    p_ndc.x = p.x * 2.0 / (window_size.x + (2 * offset)) - 1.0f;
    p_ndc.y = 1.0 - 2.0 * p.y / (window_size.y + (2 * offset));

    float p_dist = p_ndc.x * p_ndc.x + p_ndc.y * p_ndc.y;

    if (p_dist <= 1.f) {
        p_ndc.z = glm::sqrt(1.f - p_dist);
    }

    return p_ndc;
}

glm::quat ObjectEditWindow::compute_arcball_quat(glm::vec2 v0, glm::vec2 v1)
{
    glm::vec3 v0_s = compute_unit_sphere_point(v0);
    glm::vec3 v1_s = compute_unit_sphere_point(v1);

    bool v0_inbounds = (v0_s.x < 1 && v0_s.x > -1 && v0_s.y < 1 && v0_s.y > -1);
    bool v1_inbounds = (v1_s.x < 1 && v1_s.x > -1 && v1_s.y < 1 && v1_s.y > -1);

    if (v0_s != v1_s && v0_inbounds && v1_inbounds) {

        float rotation_speed = glm::clamp(camera->get_zoom() * 0.05, 0.0001, 0.01);

        glm::vec3 n0 = glm::normalize(-v0_s);
        glm::vec3 n1 = glm::normalize(-v1_s);

        float arc_length = acos(glm::min(1.f, glm::dot(n0, n1)));

        float angle = glm::min(1.f, glm::dot(v0_s, v1_s)) * arc_length;
        glm::vec3 axis = glm::normalize(glm::cross(v0_s, v1_s));

        glm::quat q = glm::angleAxis(angle, axis);
        float len = glm::length(q);

        return glm::angleAxis(angle, axis);
    }
    else {
        return glm::quat(1,0,0,0);
    }
}
