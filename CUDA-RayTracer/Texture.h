#pragma once

#include "GLCommon.h"

#include <string>
#include <iostream>
#include <assimp/texture.h>

using namespace std;

class Texture {
public:
	unsigned int id;
	string name;
	string filepath;

	GLenum target;
	GLenum type;
	GLenum format;
	GLint internalformat;

	GLint wrap;
	GLint min_filter;
	GLint mag_filter;

	int width;
	int height;
	int depth;
	int nrComponents;

	unsigned char* data; // TODO: create a HDR texture class

	Texture();
	Texture(string const& path, GLenum target, bool flip = false, bool mipmap = false, bool gamma = false, string name = "", GLint wrap = GL_REPEAT, GLint mag_filter = GL_LINEAR, GLint min_filter = GL_LINEAR);
	Texture(const aiTexture* tex, GLenum target, bool flip = false, bool mipmap = false, bool gamma = false, string name = "", GLint wrap = GL_REPEAT, GLint mag_filter = GL_LINEAR, GLint min_filter = GL_LINEAR);

	unsigned int getID() { return id; }

protected:
	unsigned int load(string filename, GLenum target);
	unsigned int loadHDR(string filename, GLenum target);
	unsigned int loadEmbedded(const aiTexture* tex);
};