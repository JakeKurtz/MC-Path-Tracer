#include "RenderEngine.h"
//#include "Film.cuh"

RenderEngine::RenderEngine()
{
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    (this)->path_tracer = std::make_shared<PathTracer>();
    (this)->raster = std::make_shared<Rasterizer>(mode->width, mode->height);
    (this)->overlay = std::make_shared<Overlay>(mode->width, mode->height);
}

void RenderEngine::render(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film, RenderMode render_mode, bool overlay_enabled)
{
    switch (render_mode) {
    case mode_rasterizer:
        raster->draw_scene(s, camera, film, 0);
        break;
    case mode_path_trace:
        path_tracer->render_image(s, camera, film);
        break;
    }

    if (overlay_enabled) 
    {
        overlay->draw(s, camera, film);
    }
}
