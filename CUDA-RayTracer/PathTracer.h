#pragma once

#include "CudaHelpers.h"

#include "Interop.h"
#include <iostream>
#include "Wavefront.cuh"
#include "Scene.h"
#include "Camera.h"
#include "Film.h"

class PathTracer
{
public:
    PathTracer();

    GLuint render_image(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);
    GLuint render_image_tile(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera);
    GLuint render_image_preview(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera);
    GLuint draw_debug(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film);

    void clear_queues();
    void set_samples(int samples);
    int get_samples();

private:
    uint32_t scr_width;
    uint32_t scr_height;

    uint32_t samples = 256;
    uint32_t max_path_length = 2; 

    uint32_t max_path_pool_size;
    uint32_t passes = 1;

    int preview_base_size = 128;
    int preview_current_size = preview_base_size;
    int preview_samples = 1;
    int preview_passes = 5;

    clock_t render_start_time;
    clock_t render_stop_time;
    bool print_time = true;

    Queues* queues;

    Interop* interop;
    cudaStream_t stream;
    cudaEvent_t  event;

    void init_interop();
    void init_queues();
};

