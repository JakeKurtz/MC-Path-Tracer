#pragma once
#include <string>
#include "cuda_runtime.h"

class Texture;

cudaTextureObject_t load_texture_uchar(Texture* tex);
cudaTextureObject_t load_texture_uchar(std::string filename);
cudaTextureObject_t load_texture_float(std::string filename);
cudaSurfaceObject_t create_surface_float(int width, int height, int nrComponents);
void destroy_texture(cudaTextureObject_t texture);
void destroy_surface(cudaSurfaceObject_t surface);