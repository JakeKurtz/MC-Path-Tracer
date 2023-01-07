#include "Interop.h"
#include <iostream>

Interop::Interop(const bool _multi_gpu, const int _fbo_count)
{
    if (GLFW_INIT) {
        multi_gpu = _multi_gpu;
        count = _fbo_count;
        index = 0;

        glGenFramebuffers(1, &fbo_main);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);

        glGenTextures(1, &col_tex);
        glBindTexture(GL_TEXTURE_2D, col_tex);

        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col_tex, 0);

        glBindTexture(GL_TEXTURE_2D, 0);

        // allocate arrays
        fbo_list = (GLuint*)calloc(count, sizeof(GLuint));
        rbo_list = (GLuint*)calloc(count, sizeof(GLuint));
        cgr_list = (cudaGraphicsResource_t*)calloc(count, sizeof(cudaGraphicsResource_t));
        ca_list = (cudaArray_t*)calloc(count, sizeof(cudaArray_t));

        // render buffer object w/a color buffer
        glCreateRenderbuffers(count, rbo_list);

        // frame buffer object
        glCreateFramebuffers(count, fbo_list);

        // attach rbo to fbo
        for (int index = 0; index < count; index++)
        {
            glNamedFramebufferRenderbuffer(fbo_list[index], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_list[index]);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    else {
        throw std::runtime_error("ERROR: glfw was not initalized. Creating Interop failed.");
    }
}

Interop::~Interop()
{
    // unregister CUDA resources
    for (int index = 0; index < count; index++)
    {
        if (cgr_list[index] != NULL)
            cudaGraphicsUnregisterResource(cgr_list[index]);
    }

    // delete rbo's
    glDeleteRenderbuffers(count, rbo_list);

    // delete fbo's
    glDeleteFramebuffers(count, fbo_list);

    // free buffers and resources
    free(fbo_list);
    free(rbo_list);
    free(cgr_list);
    free(ca_list);
}

cudaError_t Interop::set_size(const int _width, const int _height)
{
    cudaError_t cuda_err = cudaSuccess;

    // save new size
    width = _width;
    height = _height;

    glBindTexture(GL_TEXTURE_2D, col_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // resize color buffer
    for (int index = 0; index < count; index++)
    {
        // unregister resource
        if (cgr_list[index] != NULL)
            cuda_err = cudaGraphicsUnregisterResource(cgr_list[index]);

        // resize rbo
        glNamedRenderbufferStorage(rbo_list[index], GL_RGBA8, width, height);

        // probe fbo status
        // glCheckNamedFramebufferStatus(fb[index],0);

        // register rbo
        cuda_err = cudaGraphicsGLRegisterImage(&cgr_list[index],
            rbo_list[index],
            GL_RENDERBUFFER,
            cudaGraphicsRegisterFlagsSurfaceLoadStore |
            cudaGraphicsRegisterFlagsWriteDiscard);
    }

    // map graphics resources
    cuda_err = cudaGraphicsMapResources(count, cgr_list, 0);

    // get CUDA Array refernces
    for (int index = 0; index < count; index++)
    {
        cuda_err = cudaGraphicsSubResourceGetMappedArray(&ca_list[index], cgr_list[index], 0, 0);
    }

    // unmap graphics resources
    cuda_err = cudaGraphicsUnmapResources(count, cgr_list, 0);

    return cuda_err;
}

cudaError_t Interop::set_size_quick(const int _width, const int _height)
{
    cudaError_t cuda_err = cudaSuccess;

    // save new size
    width = _width;
    height = _height;

    glBindTexture(GL_TEXTURE_2D, col_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    return cuda_err;
}

void Interop::get_size(int* const _width, int* const _height)
{
    *_width = width;
    *_height = height;
}

cudaError_t Interop::map(cudaStream_t stream)
{
    if (!multi_gpu)
        return cudaSuccess;

    // map graphics resources
    return cudaGraphicsMapResources(1, &cgr_list[index], stream);
}

cudaError_t Interop::unmap(cudaStream_t stream)
{
    if (!multi_gpu)
        return cudaSuccess;

    return cudaGraphicsUnmapResources(1, &cgr_list[index], stream);
}

cudaError_t Interop::array_map()
{
    //
    // FIXME -- IS THIS EVEN NEEDED?
    // uhh... I have no idea lol.
    //

    cudaError_t cuda_err;

    // get a CUDA Array
    cuda_err = cudaGraphicsSubResourceGetMappedArray(
        &ca_list[index],
        cgr_list[index],
        0, 0
    );
    return cuda_err;
}

cudaArray_const_t Interop::array_get()
{
    return ca_list[index];
}

void Interop::swap()
{
    index = (index + 1) % count;
}

void Interop::clear()
{
    /*
    static const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };
    glInvalidateNamedFramebufferData(fb[index],1,attachments);
    */

    const GLfloat clear_color[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearNamedFramebufferfv(fbo_list[index], GL_COLOR, 0, clear_color);
    glClearNamedFramebufferfv(fbo_main, GL_COLOR, 0, clear_color);
}

void Interop::blit()
{
    glBlitNamedFramebuffer(
        fbo_list[index], fbo_main,
        0, 0, width, height,
        0, height, width, 0,
        GL_COLOR_BUFFER_BIT,
        GL_NEAREST
    );
}