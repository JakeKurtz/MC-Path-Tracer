#pragma once
#include <boost/smart_ptr/shared_ptr.hpp>
#include "Rasterizer.h"
#include "PathTracer.h"

enum RenderMode {
	mode_rasterizer,
	mode_path_trace
};

class RenderEngine
{
public:
	RenderEngine();

	void render(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film, RenderMode render_mode, bool overlay_enabled);

//private:
	std::shared_ptr<Rasterizer> raster = nullptr;		// OpenGL render pipeline
	std::shared_ptr<PathTracer> path_tracer = nullptr;	// CUDA Monte Carlo path tracer
	std::shared_ptr<Overlay> overlay = nullptr;			// OpenGL overlay
};

