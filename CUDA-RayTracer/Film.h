#pragma once
#include "GLCommon.h"

#include <string>
#include "Shader.h"
#include "globals.h"

#include <cuda_runtime.h>

#include "dRay.cuh"

#include "Observer.h"

#include <vector>
#include <numeric>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

struct Paths;

struct dFilm
{
    uint32_t width = 0;		    // horizontal image resolution
    uint32_t height = 0;	    // vertical image resolution

    uint32_t tile_width = 0;
    uint32_t tile_height = 0;

    uint32_t tile_x_pos = 0;
    uint32_t tile_y_pos = 0;

    float exposure_time = 1.f;
    float gamma = 1.f;			// monitor gamma factor
    float inv_gamma = 1.f;		// one over gamma

    uint32_t nmb_samples = 16;
    uint32_t max_path_length = 2;

    Paths* paths;
    uint32_t* samples;
    jek::Vec3f* Ld;
};

class Film : public Observer
{
public:
    Film();

    void set_exposure(const float exposure);
    void set_size(const uint32_t width, const uint32_t height);
    void set_tile_size(const uint32_t width, const uint32_t height);

    float get_exposure() const;
    void get_size(uint32_t& width, uint32_t& height) const;
    void get_tile_size(uint32_t& width, uint32_t& height) const;

    GLuint get_image_tex() const;
    uint32_t get_id() const;
    dFilm* get_dptr() const;

    void clear();
    void copy_texture_to_film(const GLuint src_tex);

    void update_tile_position();

    void update(const std::string& msg);

private:
    uint32_t id;
    std::string name;

    dFilm* dptr = nullptr;

    uint32_t pathpool_size = 0;

    GLuint image_tex = -1;

    uint32_t width = 0;
    uint32_t height = 0;

    uint32_t nmb_samples = 16;
    uint32_t max_path_length = 2;

    uint32_t tile_width = 0;
    uint32_t tile_height = 0;

    uint32_t tile_id = 0;

    uint32_t nmb_tiles = 0;
    uint32_t nmb_tile_cols = 0;
    uint32_t nmb_tile_rows = 0;

    uint32_t tile_x_pos = 0;
    uint32_t tile_y_pos = 0;

    float exposure = 1.f;

    void init_film_tex();
    void init_dptr();

    void update_tile_info();

    void update_tex_size();
    void update_dptr_size();

    void update_path_size();
};