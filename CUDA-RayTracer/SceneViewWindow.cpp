#include "SceneViewWindow.h"

SceneViewWindow::SceneViewWindow(shared_ptr<Scene> scene, std::shared_ptr<RenderEngine> render_engine) : RenderWindow(render_engine)
{
    (this)->name = "Scene View##"+ to_string(gen_id());

    (this)->scene = scene;

    init_scene();
}

void SceneViewWindow::render()
{
    render_engine->render(scene, camera, film, (RenderMode)render_mode, enable_overlay);
}

void SceneViewWindow::init_scene()
{
    film = std::make_shared<Film>();
    camera = std::make_shared<PerspectiveCamera>(glm::vec3(2.15681219, 6.10990477, -5.12322807), glm::radians(45.f), 0.01f, 10000.f);
    camera->set_yaw_pitch(-244.401138, -17.8000107);
    camera->attach(film);
    scene->attach(film);
}

void SceneViewWindow::window_size_callback(GLFWwindow* window, int width, int height)
{
}

void SceneViewWindow::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (first_mouse)
    {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }

    float xoffset = xpos - last_x;
    float yoffset = last_y - ypos; // reversed since y-coordinates go from bottom to top

    last_x = xpos;
    last_y = ypos;

    if (mouse_button_3_down && focused()) {
        camera->rotate(xoffset, yoffset);
    }

    if (mouse_button_3_down && focused()) {
        image_pan_x += xoffset * (1.f / image_scale);
        image_pan_y -= yoffset * (1.f / image_scale);
    }
}

void SceneViewWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouse_button_l_down = true;
        }
        else if (action == GLFW_RELEASE) {
            mouse_button_l_down = false;
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
}

void SceneViewWindow::mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (focused()) {
        raw_mouse_scroll -= (float)yoffset;
        if (raw_mouse_scroll < 1.f) raw_mouse_scroll = 1.f;
        if (raw_mouse_scroll > 100.f) raw_mouse_scroll = 100.f;
    }
}

void SceneViewWindow::key_callback()
{
}

void SceneViewWindow::process_input(GLFWwindow* window)
{
    //Camera controls
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->move(FORWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->move(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->move(LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->move(RIGHT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera->move(UP, delta_time);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera->move(DOWN, delta_time);
}
