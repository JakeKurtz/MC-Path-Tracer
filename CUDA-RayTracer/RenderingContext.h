#pragma once
#include <stb_image_write.h>
#include <thread>
#include "../imgui_docking/imgui.h"

#include "Camera.h"
#include "PerspectiveCamera.h"
#include "Scene.h"
#include "Rasterizer.h"
#include "PathTracer.h"

const int RASTER_MODE = 0;
const int PATHTRACE_MODE = 1;
const int DEBUG_MODE = 2;
const int WIRE_MODE = 3;

enum class RenderMode { Raster, PathTrace };

class RenderingContext
{
public:

	RenderingContext();
	RenderingContext(int width, int height);

	virtual void draw() = 0;
	virtual void init_scene() = 0;

	virtual void window_size_callback(GLFWwindow* window, int width, int height) = 0;
	virtual void mouse_callback(GLFWwindow* window, double xpos, double ypos) = 0;
	virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) = 0;
	virtual void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) = 0;
	virtual void key_callback() = 0;
	virtual void process_input(GLFWwindow* window) = 0;

	GLuint render_scene(int render_mode);

	void raster_props();
	void pathtracing_props();

	void render_properties();

	void set_width(int width);
	void set_height(int height);

	bool focused();

	//void set_scene(Scene* s);
	//void set_dscene(std::shared_ptr<dScene> ds);
	//void set_camera(Camera* c);

	std::shared_ptr<Scene> get_scene();
	//std::shared_ptr<dScene> get_dscene();
	std::shared_ptr<Camera> get_camera();

	std::shared_ptr<Rasterizer> raster = nullptr;		// OpenGL render pipeline
	std::shared_ptr<PathTracer> path_tracer = nullptr;	// CUDA Monte Carlo path tracer

	std::shared_ptr<Overlay> overlay = nullptr;			// OpenGL overlay

	//std::shared_ptr<dScene> ds;							// Scene data loaded on GPU

protected:

	ImGuiWindowFlags window_flags = 0;

	std::string window_name;

	int width;
	int height;

	ImVec2 window_size;
	ImVec2 window_min_pos;
	ImVec2 window_max_pos;
	ImVec2 window_center;

	bool shift_down = false;
	bool currently_interacting = false;
	bool buffer_reset = false;

	bool launch_draw_thread = true;

	GLuint fbo_tex = 0;

	GLuint image_texture;
	char* image_buffer;

	//std::shared_ptr<Camera> camera;	
	float camera_zoom = 1.f;

	std::shared_ptr<Scene> s;							// Scene data
	//std::shared_ptr<dScene> ds;							// Scene data loaded on GPU

	int render_mode = RASTER_MODE;
	int integrator_mode = 0;
	int raster_output_type = 0;

	float delta_time = 0.f;
	float last_frame_time = 0.f;
	float current_frame_time = 0.f;

	float raw_mouse_scroll = 1;
	float image_scale = 1.f;
	float image_pan_x = 0;
	float image_pan_y = 0;

	float last_x;
	float last_y;
	bool firstMouse = true;
	bool mouseDown = false;
	bool mouse_button_3_down = false;
	bool mouse_drag = false;
	bool click = false;

	bool enable_overlay = false;
	bool enable_shadows = true;

	void frame_time();

	void begin_imGUI();
	void end_imGUI();

	void init_image_buffer();

	void update_window_properties();
	void draw_frame_to_window();
	void bind_fbo_to_frame(GLuint frame_buffer_texture);
	void draw_menu_bar();

	void update_context_interaction_state();
};



