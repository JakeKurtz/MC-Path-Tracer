#pragma once

#include "Texture.h"

class TextureArray : public Texture
{
public:
    int layerCount;
    TextureArray(int width, int height, int layerCount, bool mipmap = false, bool gamma = false, string _name = "", GLint _wrap = GL_REPEAT, GLint _mag_filter = GL_LINEAR, GLint _min_filter = GL_LINEAR);
};


