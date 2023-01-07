#include "TextureArray.h"

TextureArray::TextureArray(int _width, int _height, int _layerCount, bool mipmap, bool gamma, string _name, GLint _wrap, GLint _mag_filter, GLint _min_filter)
{
    width = _width;
    height = _height;
    layerCount = _layerCount;

    filepath = "none";
    target = GL_TEXTURE_2D_ARRAY;
    name = _name;
    wrap = _wrap;
    mag_filter = _mag_filter;
    min_filter = _min_filter;

    glGenTextures(1, &id);
    glBindTexture(target, id);

    glTexImage3D(target, 0, GL_RG32F, width, height, layerCount, 0, GL_RG, GL_FLOAT, nullptr);
    //glTexImage3D(target, 0, GL_RG, width, height, layerCount, 0, GL_RG16, GL_FLOAT, nullptr);
    //glTexSubImage3D(target, 0, 0, 0, 0, width, height, layerCount, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (mipmap) glGenerateMipmap(target);
}