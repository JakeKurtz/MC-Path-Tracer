#pragma once 

#include "GLCommon.h"
#include <iostream>

struct Attachment 
{
public:
    Attachment(int width, int height, GLenum attachment_type, GLint internal_format, GLenum format, GLenum type) {
        (this)->width = width;
        (this)->height = height;
        (this)->attachment_type = attachment_type;
        (this)->internal_format = internal_format;
        (this)->format = format;
        (this)->type = type;

        (this)->init();
    };

    ~Attachment()
    {
        glDeleteTextures(1, &id);
    };

    void bind()
    {
        glBindTexture(GL_TEXTURE_2D, id);
    };

    void unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    };

    unsigned int get_id()
    {
        return id;
    };

    void set_size(GLsizei width, GLsizei height) 
    {
        (this)->width = width;
        (this)->height = height;

        resize();
    }

private:
    unsigned int id;
    GLsizei width;
    GLsizei height;

    GLenum attachment_type;
    GLint internal_format;
    GLenum format;
    GLenum type;

    void init()
    {
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);

        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, NULL);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        //glGenerateMipmap(GL_TEXTURE_2D);

        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment_type, GL_TEXTURE_2D, id, 0);

        glBindTexture(GL_TEXTURE_2D, 0);
    };

    void resize()
    {
        glBindTexture(GL_TEXTURE_2D, id);
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    };
};
