#include "PathTracer.h"
#include "wavefront_kernels.cuh"
#include "massert.h"

#define DEBUG

PathTracer::PathTracer()
{
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    scr_width = mode->width;
    scr_height = mode->height;

    max_path_pool_size = scr_width * scr_height;
    
    init_interop();
    init_queues();
}

GLuint PathTracer::render_image_preview(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera)
{
    /*
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    int _passes = preview_passes;

    if (*nmb_completed_paths >= nmb_paths) {
        image_complete = true;
    }

    if (s->get_dptr().update_scene()) {
        preview_current_size = preview_base_size;
        scale_film(preview_base_size);
        set_path_sample_rate(preview_samples);
        clear_buffer();
    }

    if (preview_current_size == scr_width) {
        _passes = passes;
    }

    if (image_complete && preview_current_size != scr_width) {
        upscale_current_preview_size(2);
        scale_film(preview_current_size);
        set_path_sample_rate(preview_samples);
        clear_buffer();
    }

    if (preview_current_size == scr_width) {
        set_path_sample_rate(samples);
    }

    if (!image_complete) {
        checkCudaErrors(interop->map(stream));
        for (int i = 0; i < _passes; i++) {
            wavefront_pathtrace(paths, queues, s->get_dptr().render_data, camera->get_dptr(), nmb_paths, max_path_length, interop->array_get(), event, stream, nmb_completed_paths);
            clear_queues();
        }

        construct_image(paths, s->get_dptr().render_data, camera->get_dptr(), nmb_paths, interop->array_get(), event, stream, nmb_completed_paths);

        checkCudaErrors(interop->unmap(stream));
    }
    */
    interop->blit();
   
    return interop->col_tex;
}
GLuint PathTracer::render_image_tile(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera)
{
    /*
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    if (tile_complete()) {
        checkCudaErrors(interop->map(stream));
        construct_image(paths, s->get_dptr().render_data, camera->get_dptr(), nmb_paths, interop->array_get(), event, stream, nmb_completed_paths);
        checkCudaErrors(interop->unmap(stream));
        
        if (tile_id == nmb_tiles) {
            image_complete = true;
        }
        else {
            update_tile_position();
        }
    }

    if (image_complete && print_time) {
        render_stop_time = clock();
        double timer_seconds = ((double)(render_stop_time - render_start_time)) / CLOCKS_PER_SEC;
        std::cerr << "render took: " << timer_seconds << "s" << endl;
        print_time = false;
    }

    if (s->get_dptr().update_scene()) {
        reset_tiles();
        clear_buffer();
        print_time = true;
        render_start_time = clock();
    }

    if (!image_complete) {
        checkCudaErrors(interop->map(stream));
        wavefront_pathtrace(paths, queues, s->get_dptr().render_data, camera->get_dptr(), nmb_paths, max_path_length, interop->array_get(), event, stream, nmb_completed_paths);
        clear_queues();
        checkCudaErrors(interop->unmap(stream));
    }

    interop->blit();
    */
    return interop->col_tex;
}
GLuint PathTracer::render_image(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t film_width, film_height;
    film->get_size(film_width, film_height);
    
    interop->set_size_quick(film_width, film_height);
    //interop->clear();
    clear_queues();

    checkCudaErrors(interop->map(stream));
    wavefront_pathtrace(scene, camera, film, queues, interop->array_get(), event, stream);
    checkCudaErrors(interop->unmap(stream));

    interop->blit();

    film->copy_texture_to_film(interop->col_tex);
    film->update_tile_position();

    return interop->col_tex;
}
GLuint PathTracer::draw_debug(std::shared_ptr<Scene> s, std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    uint32_t width, height;
    film->get_size(width, height);
    interop->set_size_quick(width, height);

    checkCudaErrors(interop->map(stream));
    debug_raytracer(s, camera, film, interop->array_get(), event, stream);
    checkCudaErrors(interop->unmap(stream));

    interop->blit();
    film->copy_texture_to_film(interop->col_tex);
    film->update_tile_position();

    return interop->col_tex;
}

void PathTracer::clear_queues()
{
    queues->new_path_len = 0;
    queues->ext_path_len = 0;
    queues->shadow_len = 0;
    queues->mat_mix_len = 0;
}
void PathTracer::set_samples(int samples)
{
    //(this)->samples = samples;
    //set_path_sample_rate(samples);
}
int PathTracer::get_samples()
{
    return 0;// samples;
}

void PathTracer::init_interop()
{
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));// | cudaStreamNonBlocking));   // optionally ignore default stream behavior
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventDefault)); // | cudaEventDisableTiming);

    // Testing: DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
    try {
        interop = new Interop(true, 2);
        checkCudaErrors(interop->set_size(scr_width, scr_height));
        //interop_initialized = true;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
void PathTracer::init_queues()
{
    checkCudaErrors(cudaMallocManaged(&queues, sizeof(Queues)));
    checkCudaErrors(cudaMallocManaged(&queues->new_path, sizeof(uint32_t) * max_path_pool_size));
    checkCudaErrors(cudaMallocManaged(&queues->ext_path, sizeof(uint32_t) * max_path_pool_size));
    checkCudaErrors(cudaMallocManaged(&queues->shadow_path, sizeof(uint32_t) * max_path_pool_size));
    checkCudaErrors(cudaMallocManaged(&queues->mat_mix_path, sizeof(uint32_t) * max_path_pool_size));
}