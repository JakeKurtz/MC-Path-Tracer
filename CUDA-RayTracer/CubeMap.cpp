#include "CubeMap.h"

CubeMap::CubeMap(int size, bool mipmap, bool gamma, string _name, GLint _wrap, GLint _mag_filter, GLint _min_filter)
{
    width = size;
    height = size;

    filepath = "none";
    target = GL_TEXTURE_CUBE_MAP;
    name = _name;
    wrap = _wrap;
    mag_filter = _mag_filter;
    min_filter = _min_filter;

    glGenTextures(1, &id);
    glBindTexture(target, id);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, size, size, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_R, wrap);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter);

    if (mipmap) glGenerateMipmap(target);
}