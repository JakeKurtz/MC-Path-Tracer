/***************************************************************************************
    *    Title: A tiny example of CUDA + OpenGL interop with write-only surfaces and CUDA kernels. Uses GLFW+GLAD.
    *    Author: Allan MacKinnon
    *    Availability: https://gist.github.com/allanmac/4ff11985c3562830989f
    *
***************************************************************************************/

#pragma once

#include "GLCommon.h"
#include "globals.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_gl_interop.h>
#include "FrameBuffer.h"

class Interop
{
private:
    bool multi_gpu;     // split GPUs?

    // number of fbo's
    int count;
    int index;

    int width;
    int height;

    // CUDA resources
    cudaGraphicsResource_t* cgr_list;
    cudaArray_t* ca_list;

public:

    // GL buffers
    GLuint fbo_main;
    GLuint* fbo_list;
    GLuint* rbo_list;

    GLuint col_tex;

    Interop(const bool multi_gpu, const int fbo_count);
    ~Interop();

    cudaError_t set_size(const int width, const int height);
    cudaError_t set_size_quick(const int width, const int height);
    void get_size(int* const width, int* const height);
    cudaError_t map(cudaStream_t stream);
    cudaError_t unmap(cudaStream_t stream);
    cudaError_t array_map();
    cudaArray_const_t array_get();
    void swap();
    void clear();
    void blit();
};

