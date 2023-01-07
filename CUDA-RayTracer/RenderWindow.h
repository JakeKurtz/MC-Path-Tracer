#pragma once
#include <stb_image_write.h>
#include <thread>
#include "../imgui_docking/imgui.h"

#include "Camera.h"
#include "PerspectiveCamera.h"
#include "Scene.h"
#include "Rasterizer.h"
#include "PathTracer.h"

#include "Window.h"
#include "RenderEngine.h"
//#include "Film.cuh"

class RenderWindow : public Window
{
public:

	RenderWindow(std::shared_ptr<RenderEngine> render_engine);

	void draw();

	virtual void init_scene() = 0;
	virtual void render() = 0;

	std::shared_ptr<Scene> get_scene();
	std::shared_ptr<Camera> get_camera();
	std::shared_ptr<Film> get_film();

protected:
	std::shared_ptr<RenderEngine> render_engine;

	std::shared_ptr<Scene> scene;
	std::shared_ptr<Camera> camera;
	std::shared_ptr<Film> film;

	int render_mode = 0;
	int integrator_mode = 0;
	int raster_output_type = 0;

	bool enable_overlay = false;
	bool enable_shadows = true;

	void draw_frame_to_window();
	void menu_bar();
};



