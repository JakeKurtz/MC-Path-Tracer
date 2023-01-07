#pragma once

#include "RenderWindow.h"
#include "RenderObject.h"
#include "Observer.h"

const int OBJ_TYPE_CURVE = 0;
const int OBJ_TYPE_MODEL = 1;

class ObjectEditWindow : public RenderWindow
{
public:
	ObjectEditWindow(std::shared_ptr<RenderEngine> render_engine);

	void render();
	void init_scene();

	//void set_curve(shared_ptr<Curve> curve);
	void set_obj(shared_ptr<RenderObject> obj);

	void window_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void key_callback();
	void process_input(GLFWwindow* window);

	//void update(const std::string& msg, const SceneUpdateFlags flag);

private:
	shared_ptr<Transform> transform_edit;
	shared_ptr<Transform> transform_world;
	shared_ptr<Curve> curve = nullptr;
	shared_ptr<Mesh> mesh = nullptr;

	shared_ptr<EnvironmentLight> env_light_color;
	shared_ptr<EnvironmentLight> env_light_hrd;

	shared_ptr<RenderObject> r_obj = nullptr;

	glm::vec3 centroid = glm::vec3(0.f);

	int object_type = OBJ_TYPE_CURVE;

	float yaw = 0.f;
	float pitch = 0.f;

	glm::vec3 compute_unit_sphere_point(glm::vec2 p);
	glm::quat compute_arcball_quat(glm::vec2 v0, glm::vec2 v1);
};

