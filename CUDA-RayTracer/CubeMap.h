#pragma once

#include "Texture.h"
#include <vector>

class CubeMap : public Texture
{
public:
	CubeMap(int size, bool mipmap = false, bool gamma = false, std::string _name = "", GLint _wrap = GL_REPEAT, GLint _mag_filter = GL_LINEAR, GLint _min_filter = GL_LINEAR);
};

