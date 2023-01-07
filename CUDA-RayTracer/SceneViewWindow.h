#pragma once

#include "RenderWindow.h"

class SceneViewWindow : public RenderWindow
{
public:
    SceneViewWindow(shared_ptr<Scene> scene, std::shared_ptr<RenderEngine> render_engine);

	void render();
	void init_scene();

	void window_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypo);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void key_callback();
	void process_input(GLFWwindow* window);
};

