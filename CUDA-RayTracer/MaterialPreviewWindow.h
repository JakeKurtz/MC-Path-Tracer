#pragma once

#include "RenderWindow.h"
#include "RenderObject.h"

class MaterialPreviewWindow : public RenderWindow
{
public:
	MaterialPreviewWindow(int width, int height, std::shared_ptr<RenderEngine> render_engine);

	void init_scene();

	void draw_windowless();
	void render();

	std::shared_ptr<Material> get_material();
	void set_material(shared_ptr<Material> material);

	void window_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void key_callback();
	void process_input(GLFWwindow* window);

private:
	shared_ptr<Transform> transform_edit;
	shared_ptr<Transform> transform_world;

	std::shared_ptr<EnvironmentLight> env_light_hrd;

	shared_ptr<RenderObject> r_obj = nullptr;

	glm::vec3 centroid = glm::vec3(0.f);

	float yaw = 0.f;
	float pitch = 0.f;
};

